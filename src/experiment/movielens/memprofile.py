"""Memory-profile the movielens producer and runner in isolation.

Launches each phase as a separate subprocess (so their RSS can't pollute
each other) and samples peak RSS via `ps`. Useful for verifying that the
runner's footprint stays small while the producer carries the ml-32m load.

Typical run:

    python -m experiment.movielens.memprofile \\
      --config src/experiment/movielens/100000_s44_config.json \\
      --sample-idx 0

Use ``--load-only`` (default) to skip GEPA/LM calls and only measure the
cost of loading shards into dspy.Example form. Drop ``--load-only`` to
measure full baseline + GEPA (requires a reachable LM).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple


def _sample_rss(pid: int, samples: List[Tuple[float, int]],
                interval_s: float, stop: threading.Event) -> None:
    """Background thread: record (timestamp, rss_kb) until the process exits."""
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True, text=True, check=False,
            )
            line = r.stdout.strip()
            if not line:
                return
            samples.append((time.time(), int(line.split()[0])))
        except Exception:
            return
        stop.wait(interval_s)


def run_with_sampling(cmd: List[str], interval_s: float = 0.25,
                      label: str = "") -> dict:
    """Run `cmd` as a subprocess while sampling RSS periodically."""
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    samples: List[Tuple[float, int]] = []
    stop = threading.Event()
    t = threading.Thread(
        target=_sample_rss, args=(proc.pid, samples, interval_s, stop),
        daemon=True,
    )
    t.start()
    exit_code = proc.wait()
    stop.set()
    t.join(timeout=interval_s * 4)
    elapsed = time.time() - t0

    if not samples:
        return {
            "label": label, "elapsed_s": elapsed, "exit_code": exit_code,
            "peak_mb": None, "final_mb": None, "n_samples": 0,
        }

    peak_kb = max(kb for _, kb in samples)
    final_kb = samples[-1][1]
    return {
        "label": label,
        "elapsed_s": elapsed,
        "exit_code": exit_code,
        "peak_mb": peak_kb / 1024.0,
        "final_mb": final_kb / 1024.0,
        "n_samples": len(samples),
        "samples": samples,
    }


def _fmt_mb(mb: Optional[float]) -> str:
    return "   n/a   " if mb is None else f"{mb:8.1f} MB"


def _print_phase(report: dict) -> None:
    print(f"  elapsed  : {report['elapsed_s']:.1f}s")
    print(f"  peak RSS : {_fmt_mb(report['peak_mb'])}")
    print(f"  final RSS: {_fmt_mb(report['final_mb'])}")
    print(f"  samples  : {report['n_samples']}")
    print(f"  exit code: {report['exit_code']}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-profile movielens producer + runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to the SetupConfig JSON")
    parser.add_argument("--out-dir", default="/tmp/ml_memprofile",
                        help="Shard output directory (wiped first)")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Cohort index to produce + run")
    parser.add_argument("--interval", type=float, default=0.25,
                        help="RSS sampling interval in seconds")
    parser.add_argument("--load-only", action="store_true", default=True,
                        help="Runner phase stops after loading shards (no LM calls)")
    parser.add_argument("--with-gepa", dest="load_only", action="store_false",
                        help="Runner phase runs full baseline + GEPA (needs LM)")
    parser.add_argument("--model", default="openai/qwen3-235b")
    parser.add_argument("--api-base", default="http://localhost:4000")
    parser.add_argument("--api-key", default="noop")
    parser.add_argument("--skip-run", action="store_true",
                        help="Only profile the producer phase")
    parser.add_argument("--save-trace", type=str, default=None,
                        help="Optional path to write raw (phase,timestamp,rss_kb) CSV trace")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    idx = args.sample_idx

    # --- phase 1: produce -------------------------------------------------
    print(f"=== [1/2] produce (cohort {idx}) ===", flush=True)
    produce_cmd = [
        python, "-m", "experiment.movielens.distributed_main", "produce",
        "--config", args.config,
        "--out", str(out_dir),
        "--start", str(idx),
        "--limit", "1",
    ]
    prod = run_with_sampling(produce_cmd, args.interval, label="produce")
    _print_phase(prod)

    shard_files = sorted(out_dir.glob("*.jsonl"))
    total_bytes = sum(p.stat().st_size for p in shard_files)
    print(f"  shards   : {len(shard_files)} files, {total_bytes:,} bytes")

    if prod["exit_code"]:
        return prod["exit_code"]
    if not shard_files:
        print("No shards produced (cohort likely below threshold). Aborting.")
        return 1
    if args.skip_run:
        _write_trace(args.save_trace, [prod])
        return 0

    # --- phase 2: run -----------------------------------------------------
    mode = "load-only" if args.load_only else "full GEPA"
    print(f"\n=== [2/2] run ({mode}, cohort {idx}) ===", flush=True)
    run_cmd = [
        python, "-m", "experiment.movielens.distributed_main", "run",
        "--dataset-dir", str(out_dir),
        "--sample-idx", str(idx),
    ]
    if args.load_only:
        run_cmd.append("--load-only")
    else:
        run_cmd.extend([
            "--model", args.model,
            "--api-base", args.api_base,
            "--api-key", args.api_key,
            "--out", str(out_dir / f"result_{idx}.json"),
        ])
    run = run_with_sampling(run_cmd, args.interval, label="run")
    _print_phase(run)

    # --- summary ----------------------------------------------------------
    print("\n=== summary ===")
    print(f"  producer peak: {_fmt_mb(prod['peak_mb'])}  ({prod['elapsed_s']:.1f}s)")
    print(f"  runner   peak: {_fmt_mb(run['peak_mb'])}  ({run['elapsed_s']:.1f}s)")
    if prod["peak_mb"] and run["peak_mb"]:
        ratio = prod["peak_mb"] / run["peak_mb"]
        print(f"  producer/runner peak ratio: {ratio:.1f}x")
        saved = prod["peak_mb"] - run["peak_mb"]
        print(f"  RSS saved per runner proc : {saved:,.1f} MB")

    _write_trace(args.save_trace, [prod, run])
    return run["exit_code"] or 0


def _write_trace(path: Optional[str], phases: List[dict]) -> None:
    if not path:
        return
    with open(path, "w") as f:
        f.write("phase,t_rel_s,rss_kb\n")
        for p in phases:
            samples = p.get("samples") or []
            if not samples:
                continue
            t0 = samples[0][0]
            for t, kb in samples:
                f.write(f"{p['label']},{t - t0:.3f},{kb}\n")
    print(f"  trace    : {path}")


if __name__ == "__main__":
    sys.exit(main())
