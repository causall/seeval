"""Distributed producer/consumer for the movielens GEPA experiment.

The producer walks a ``sample_results`` file one cohort at a time, writes
per-cohort dspy.Example shards to disk, and drops its intermediates between
iterations. Consumers only need the manifest + three small JSONL files per
cohort; they never touch the 32M-rating corpus.

Subcommands:
  produce   Build shards for cohorts in [--start, --end).
  run       Execute baseline + GEPA for a single cohort from a shard dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import random
import dspy

from experiment.core import setup_experiment_lm
from experiment.movielens import main as ml_main
from experiment.movielens import group_dataset as gd
from experiment.movielens import shards as shard_io
from experiment.movielens.group_dataset import load_config_from_disk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="distributed_main",
        description="Produce/consume sharded movielens experiment datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_produce = sub.add_parser(
        "produce", help="Build dspy.Example shards for a range of cohorts"
    )
    p_produce.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the SetupConfig JSON (produced by main.py --setup)",
    )
    p_produce.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output directory for shards + manifest.json",
    )
    p_produce.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive) into sample_results",
    )
    p_produce.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive). Defaults to len(sample_results)",
    )
    p_produce.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of cohorts to produce (applied after start)",
    )
    p_produce.add_argument(
        "--exp-valid-movie-count",
        type=int,
        default=None,
        help="Override the exp_valid_movie_count threshold from the config",
    )
    p_produce.add_argument(
        "--seed", type=int, default=None, help="Override the seed from the config"
    )
    p_produce.add_argument(
        "--total-examples",
        type=int,
        default=None,
        help="If set, downsample each cohort's movie_ids to exactly this many "
        "BEFORE the 80/20->80/20 split so every shard has identical sizes. "
        "Cohorts with fewer valid movies are skipped.",
    )

    p_run = sub.add_parser("run", help="Run baseline + GEPA for a single cohort shard")
    p_run.add_argument(
        "--dataset-dir",
        required=True,
        type=str,
        help="Directory containing manifest.json + shards",
    )
    p_run.add_argument(
        "--sample-idx",
        required=True,
        type=int,
        help="Cohort index to execute (must exist in the shard dir)",
    )
    p_run.add_argument("--model", type=str, default="openai/qwen3-235b")
    p_run.add_argument("--api-base", type=str, default="http://localhost:4000")
    p_run.add_argument("--api-key", type=str, default="noop")
    p_run.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write JSON result summary",
    )
    p_run.add_argument(
        "--save-program",
        type=str,
        default=None,
        help="Optional path to save the optimized program JSON",
    )
    p_run.add_argument(
        "--load-only",
        action="store_true",
        help="Load the shards into memory, print counts, and exit (for profiling)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# produce
# ---------------------------------------------------------------------------


def _resolve_range(
    total: int, start: int, end: Optional[int], limit: Optional[int]
) -> range:
    if start < 0 or start > total:
        raise ValueError(f"--start={start} out of range for {total} sample_results")
    stop = total if end is None else min(end, total)
    if limit is not None:
        stop = min(stop, start + limit)
    if stop < start:
        raise ValueError(f"Empty range: start={start} end={stop}")
    return range(start, stop)


def cmd_produce(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    setup_config = load_config_from_disk(config_path)
    if args.seed is not None:
        setup_config = setup_config.model_copy(update={"seed": args.seed})
    if args.exp_valid_movie_count is not None:
        setup_config = setup_config.model_copy(
            update={"exp_valid_movie_count": args.exp_valid_movie_count}
        )

    output_file = Path(setup_config.output_file)
    if not output_file.is_absolute() and not output_file.exists():
        # Fall back to resolving relative to the config file's directory.
        candidate = config_path.parent / setup_config.output_file
        if candidate.exists():
            output_file = candidate
    sample_results = gd.load_sample_results_from_disk(output_file)
    idx_range = _resolve_range(len(sample_results), args.start, args.end, args.limit)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Manifest-level criteria is only a template -- consumers rebuild per-cohort
    # criteria from (seed, sample_idx) via make_cohort_token at run time.
    criteria_template = ml_main.get_movie_rating_criteria("c-template")

    print(
        f"[produce] cohorts {idx_range.start}..{idx_range.stop} "
        f"(of {len(sample_results)}) -> {out_dir}",
        flush=True,
    )

    t_cache_start = time.time()
    cache = ml_main.establish_cache()
    print(f"[produce] cache ready in {time.time() - t_cache_start:.1f}s", flush=True)

    produced: List[int] = []
    skipped: List[int] = []

    for idx in idx_range:
        t0 = time.time()
        # Fresh RNG per cohort keeps output deterministic and independent of
        # how many cohorts preceded this one in the range.
        rng = random.Random(setup_config.seed + idx)
        splits = ml_main.build_split_datasets(
            cache,
            sample_results[idx],
            rng,
            setup_config.exp_valid_movie_count,
            total_examples=args.total_examples,
        )
        if splits is None:
            skipped.append(idx)
            print(f"[produce] idx={idx} skipped (below threshold)", flush=True)
            continue

        cohort_token = ml_main.make_cohort_token(setup_config.seed, idx)
        per_cohort_criteria = ml_main.get_movie_rating_criteria(cohort_token)
        examples = ml_main.build_split_examples(
            splits, per_cohort_criteria, setup_config.seed
        )

        n_train = shard_io.write_examples_shard(
            shard_io.shard_path(out_dir, "train", idx), examples.train
        )
        n_val = shard_io.write_examples_shard(
            shard_io.shard_path(out_dir, "val", idx), examples.validation
        )
        n_test = shard_io.write_examples_shard(
            shard_io.shard_path(out_dir, "test", idx), examples.test
        )

        produced.append(idx)
        print(
            f"[produce] idx={idx} train={n_train} val={n_val} test={n_test} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

        del splits, examples
        gc.collect()

    manifest = shard_io.ShardManifest(
        criteria=criteria_template,
        seed=setup_config.seed,
        produced_indices=produced,
        exp_valid_movie_count=setup_config.exp_valid_movie_count,
        total_examples=args.total_examples,
        source_config=str(config_path),
        created_at=datetime.now(timezone.utc),
    )
    shard_io.write_manifest(shard_io.manifest_path(out_dir), manifest)

    print(
        f"[produce] done. produced={len(produced)} skipped={len(skipped)} "
        f"manifest={shard_io.manifest_path(out_dir)}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.dataset_dir)
    manifest = shard_io.load_manifest(shard_io.manifest_path(out_dir))
    idx = args.sample_idx

    # Rebuild criteria from (seed, idx) so the opaque cohort anchor in rubric
    # descs matches the grader instruction. Manifest.criteria is only a template.
    cohort_token = ml_main.make_cohort_token(manifest.seed, idx)
    criteria = ml_main.get_movie_rating_criteria(cohort_token)

    if idx not in manifest.produced_indices:
        print(
            f"[run] WARNING: idx={idx} not listed in manifest.produced_indices "
            f"({manifest.produced_indices[:5]}{'...' if len(manifest.produced_indices) > 5 else ''})",
            file=sys.stderr,
        )

    train = shard_io.load_examples_shard(
        shard_io.shard_path(out_dir, "train", idx), criteria
    )
    validation = shard_io.load_examples_shard(
        shard_io.shard_path(out_dir, "val", idx), criteria
    )
    test = shard_io.load_examples_shard(
        shard_io.shard_path(out_dir, "test", idx), criteria
    )
    print(
        f"[run] idx={idx} cohort_token={cohort_token} "
        f"train={len(train)} val={len(validation)} test={len(test)}",
        flush=True,
    )

    if args.load_only:
        print("[run] --load-only set; exiting before configuring LM", flush=True)
        return 0

    lm = setup_experiment_lm(args.model, args.api_base, args.api_key)
    dspy.configure(lm=lm)

    examples = ml_main.SplitExamples(train=train, validation=validation, test=test)

    t0 = time.time()
    optimized_program, _teleprompter, baseline_score, optimized_score = (
        ml_main.run_gepa_experiment(lm, examples, criteria, cohort_token)
    )
    duration = time.time() - t0

    program_path: Optional[str] = None
    if args.save_program:
        program_path = str(Path(args.save_program))
        Path(args.save_program).parent.mkdir(parents=True, exist_ok=True)
        optimized_program.save(args.save_program)

    result = {
        "sample_idx": idx,
        "cohort_token": cohort_token,
        "model": args.model,
        "baseline_score": float(baseline_score),
        "optimized_score": float(optimized_score),
        "duration_s": duration,
        "optimized_program_path": program_path,
        "dataset_dir": str(out_dir),
        "n_train": len(train),
        "n_val": len(validation),
        "n_test": len(test),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[run] wrote result -> {out_path}", flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)

    return 0


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "produce":
        return cmd_produce(args)
    if args.cmd == "run":
        return cmd_run(args)
    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
