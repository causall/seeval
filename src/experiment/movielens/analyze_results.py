"""Analyze GEPA MovieLens shard results.

Joins per-cohort ``result_*.json`` summaries with the matching
``program_*.json`` (the GEPA-optimized grader prompt) to produce a single
DataFrame of per-cohort metrics, a printed/csv summary, and a pack of plots
(paired baseline vs optimized violins, delta histogram, quality-vs-prompt-size
scatters, duration distributions, etc.).

CLI:
    python -m experiment.movielens.analyze_results \
        --shards shards/s48_n140 shards/s44 \
        --out shards/analysis

Prompt-token counts use ``tiktoken`` (``cl100k_base``) as a proxy for
"tokens each optimized program uses" — the grader's ``signature.instructions``
string is the actual artifact GEPA produces per cohort.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:  # pragma: no cover - optional
    sns = None  # type: ignore
    _HAS_SEABORN = False

try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional
    _scipy_stats = None  # type: ignore
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Tokenizer (tiktoken, lazy + cached)
# ---------------------------------------------------------------------------

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken  # lazy import; only needed when actually counting
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class CohortRow:
    shard: str
    sample_idx: int
    cohort_token: str
    model: str
    baseline_score: float
    optimized_score: float
    delta: float
    duration_s: float
    duration_min: float
    instruction_chars: int
    instruction_tokens: int
    program_bytes: int
    n_train: int
    n_val: int
    n_test: int


def _resolve_program_path(result_path: Path, result: dict,
                          repo_root: Path) -> Optional[Path]:
    """Find the ``program_<idx>.json`` for a result, trying the recorded path
    first, then ``<shard>/program_<idx>.json`` as a fallback."""
    rec = result.get("optimized_program_path")
    if rec:
        cand = Path(rec)
        if not cand.is_absolute():
            cand = repo_root / cand
        if cand.exists():
            return cand
    idx = result.get("sample_idx")
    if idx is None:
        return None
    fallback = result_path.parent / f"program_{idx}.json"
    return fallback if fallback.exists() else None


def _extract_instruction(program: dict) -> str:
    """Pull the grader instruction string out of a saved dspy program JSON.

    The format is nested: ``<predictor_name>.signature.instructions``. We take
    the first predictor whose value is a dict holding a ``signature.instructions``
    — in this experiment that's ``grader.predict``.
    """
    for _, v in program.items():
        if isinstance(v, dict):
            sig = v.get("signature")
            if isinstance(sig, dict):
                instr = sig.get("instructions")
                if isinstance(instr, str):
                    return instr
    return ""


def load_shard(shard_dir: Path, repo_root: Path) -> List[CohortRow]:
    rows: List[CohortRow] = []
    for result_path in sorted(shard_dir.glob("result_*.json")):
        try:
            result = json.loads(result_path.read_text())
        except Exception as e:  # pragma: no cover - defensive
            print(f"[warn] failed to read {result_path}: {e}", file=sys.stderr)
            continue

        program_path = _resolve_program_path(result_path, result, repo_root)
        if program_path is None:
            print(f"[warn] no program file for {result_path}", file=sys.stderr)
            instruction = ""
            program_bytes = 0
        else:
            try:
                program = json.loads(program_path.read_text())
                instruction = _extract_instruction(program)
                program_bytes = program_path.stat().st_size
            except Exception as e:
                print(f"[warn] failed to read program {program_path}: {e}",
                      file=sys.stderr)
                instruction = ""
                program_bytes = 0

        baseline = float(result.get("baseline_score", float("nan")))
        optimized = float(result.get("optimized_score", float("nan")))
        duration_s = float(result.get("duration_s", float("nan")))
        instruction_tokens = _count_tokens(instruction) if instruction else 0

        rows.append(CohortRow(
            shard=shard_dir.name,
            sample_idx=int(result.get("sample_idx", -1)),
            cohort_token=str(result.get("cohort_token", "")),
            model=str(result.get("model", "")),
            baseline_score=baseline,
            optimized_score=optimized,
            delta=optimized - baseline,
            duration_s=duration_s,
            duration_min=duration_s / 60.0 if np.isfinite(duration_s) else float("nan"),
            instruction_chars=len(instruction),
            instruction_tokens=instruction_tokens,
            program_bytes=program_bytes,
            n_train=int(result.get("n_train", 0)),
            n_val=int(result.get("n_val", 0)),
            n_test=int(result.get("n_test", 0)),
        ))
    return rows


def build_dataframe(shards: Iterable[Path], repo_root: Path) -> pd.DataFrame:
    all_rows: List[CohortRow] = []
    for shard_dir in shards:
        rows = load_shard(shard_dir, repo_root)
        print(f"[load] {shard_dir}: {len(rows)} cohorts", flush=True)
        all_rows.extend(rows)
    if not all_rows:
        raise SystemExit("No cohorts loaded. Check --shards paths.")
    df = pd.DataFrame([r.__dict__ for r in all_rows])
    return df


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "baseline_score", "optimized_score", "delta",
        "duration_s", "duration_min",
        "instruction_chars", "instruction_tokens", "program_bytes",
        "n_train", "n_val", "n_test",
    ]
    desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["iqr"] = desc["75%"] - desc["25%"]
    return desc


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    wins = int((df["optimized_score"] > df["baseline_score"]).sum())
    ties = int((df["optimized_score"] == df["baseline_score"]).sum())
    losses = int((df["optimized_score"] < df["baseline_score"]).sum())
    mean_delta = df["delta"].mean()
    median_delta = df["delta"].median()
    print("\n=== paired baseline vs optimized ===")
    print(f"  n cohorts:        {n}")
    print(f"  wins / ties / losses: {wins} / {ties} / {losses}")
    print(f"  win-rate:         {wins / n:.1%}")
    print(f"  mean delta:       {mean_delta:+.3f}")
    print(f"  median delta:     {median_delta:+.3f}")
    if _HAS_SCIPY and n >= 2:
        t, p = _scipy_stats.ttest_rel(df["optimized_score"], df["baseline_score"])
        print(f"  paired t-test:    t={t:.3f}  p={p:.3e}")
        w, pw = _scipy_stats.wilcoxon(
            df["optimized_score"], df["baseline_score"])
        print(f"  wilcoxon:         W={w:.1f}  p={pw:.3e}")

    print("\n=== per-metric summary ===")
    print(summarize(df).round(3).to_string())

    print("\n=== correlations with optimized_score ===")
    corr_cols = ["baseline_score", "duration_s", "instruction_tokens",
                 "instruction_chars", "program_bytes"]
    corr = df[corr_cols + ["optimized_score"]].corr(method="pearson")["optimized_score"].drop("optimized_score")
    scorr = df[corr_cols + ["optimized_score"]].corr(method="spearman")["optimized_score"].drop("optimized_score")
    comb = pd.DataFrame({"pearson": corr, "spearman": scorr}).round(3)
    print(comb.to_string())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.rcParams.update({
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 110,
            "savefig.dpi": 140,
        })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _paired_violin(ax: plt.Axes, df: pd.DataFrame) -> None:
    base = df["baseline_score"].values
    opt = df["optimized_score"].values
    if _HAS_SEABORN:
        long = pd.DataFrame({
            "score": np.concatenate([base, opt]),
            "kind": (["baseline"] * len(base)) + (["optimized"] * len(opt)),
        })
        sns.violinplot(data=long, x="kind", y="score", ax=ax,
                       inner="quartile", cut=0)
        sns.stripplot(data=long, x="kind", y="score", ax=ax,
                      color="black", alpha=0.35, size=3, jitter=0.15)
    else:
        parts = ax.violinplot([base, opt], showmeans=False,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["baseline", "optimized"])
        xs_b = np.random.default_rng(0).normal(1, 0.04, size=len(base))
        xs_o = np.random.default_rng(1).normal(2, 0.04, size=len(opt))
        ax.scatter(xs_b, base, s=10, color="black", alpha=0.35)
        ax.scatter(xs_o, opt, s=10, color="black", alpha=0.35)
    for b, o in zip(base, opt):
        ax.plot([0 if _HAS_SEABORN else 1, 1 if _HAS_SEABORN else 2],
                [b, o], color="gray", alpha=0.15, linewidth=0.5)
    ax.set_title("paired baseline vs optimized")
    ax.set_ylabel("score")
    ax.set_xlabel("")


def _delta_hist(ax: plt.Axes, df: pd.DataFrame) -> None:
    deltas = df["delta"].values
    if _HAS_SEABORN:
        sns.histplot(deltas, kde=True, ax=ax, color="steelblue", bins=30)
    else:
        ax.hist(deltas, bins=30, color="steelblue", alpha=0.8,
                edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="zero")
    mu, med = np.mean(deltas), np.median(deltas)
    ax.axvline(mu, color="black", linestyle="-", linewidth=1,
               label=f"mean={mu:+.2f}")
    ax.axvline(med, color="green", linestyle=":", linewidth=1.5,
               label=f"median={med:+.2f}")
    ax.legend(fontsize=9)
    ax.set_title("delta = optimized - baseline")
    ax.set_xlabel("delta")


def _scatter_base_vs_opt(ax: plt.Axes, df: pd.DataFrame) -> None:
    for shard, sub in df.groupby("shard"):
        ax.scatter(sub["baseline_score"], sub["optimized_score"],
                   s=18, alpha=0.7, label=shard)
    lo = min(df["baseline_score"].min(), df["optimized_score"].min())
    hi = max(df["baseline_score"].max(), df["optimized_score"].max())
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--",
            linewidth=1, label="y=x")
    ax.set_xlabel("baseline_score")
    ax.set_ylabel("optimized_score")
    ax.set_title("baseline vs optimized (per cohort)")
    ax.legend(fontsize=9)


def _violin_by_shard(ax: plt.Axes, df: pd.DataFrame, col: str,
                     title: str) -> None:
    if _HAS_SEABORN:
        sns.violinplot(data=df, x="shard", y=col, ax=ax, inner="quartile",
                       cut=0)
        sns.stripplot(data=df, x="shard", y=col, ax=ax,
                      color="black", alpha=0.35, size=3, jitter=0.15)
    else:
        shards = sorted(df["shard"].unique())
        data = [df.loc[df["shard"] == s, col].dropna().values for s in shards]
        ax.violinplot(data, showmedians=True, showextrema=False)
        ax.set_xticks(range(1, len(shards) + 1))
        ax.set_xticklabels(shards)
    ax.set_title(title)
    ax.set_ylabel(col)
    ax.set_xlabel("")


def _program_size_violin(fig_axes, df: pd.DataFrame) -> None:
    ax_tok, ax_char = fig_axes
    _violin_by_shard(ax_tok, df, "instruction_tokens",
                     "program size (tokens)")
    _violin_by_shard(ax_char, df, "instruction_chars",
                     "program size (chars)")


def _scatter_with_fit(ax: plt.Axes, df: pd.DataFrame, x: str, y: str,
                      title: str) -> None:
    sub = df[[x, y, "shard"]].dropna()
    for shard, s in sub.groupby("shard"):
        ax.scatter(s[x], s[y], s=18, alpha=0.7, label=shard)
    if len(sub) >= 2:
        xs = sub[x].values.astype(float)
        ys = sub[y].values.astype(float)
        m, b = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, m * xx + b, color="black", linestyle="--", linewidth=1)
        pear = float(np.corrcoef(xs, ys)[0, 1])
        if _HAS_SCIPY:
            sp = float(_scipy_stats.spearmanr(xs, ys).statistic)
            ax.set_title(f"{title}\npearson={pear:.2f}  spearman={sp:.2f}  n={len(sub)}")
        else:
            ax.set_title(f"{title}\npearson={pear:.2f}  n={len(sub)}")
    else:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=9)


def render_plots(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    _paired_violin(ax, df)
    _save(fig, out_dir / "paired_baseline_vs_optimized.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _delta_hist(ax, df)
    _save(fig, out_dir / "delta_hist.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_base_vs_opt(ax, df)
    _save(fig, out_dir / "scatter_baseline_vs_optimized.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _violin_by_shard(ax, df, "duration_s", "optimization duration (s)")
    _save(fig, out_dir / "duration_violin.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _program_size_violin(axes, df)
    _save(fig, out_dir / "program_size_violin.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_with_fit(ax, df, "instruction_tokens", "optimized_score",
                      "quality vs prompt tokens")
    _save(fig, out_dir / "quality_vs_tokens.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_with_fit(ax, df, "instruction_tokens", "delta",
                      "lift vs prompt tokens")
    _save(fig, out_dir / "delta_vs_tokens.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_with_fit(ax, df, "duration_s", "optimized_score",
                      "optimized score vs duration")
    _save(fig, out_dir / "duration_vs_optimized.png")

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    _paired_violin(axes[0, 0], df)
    _delta_hist(axes[0, 1], df)
    _scatter_base_vs_opt(axes[0, 2], df)
    _violin_by_shard(axes[1, 0], df, "duration_s", "duration (s)")
    _violin_by_shard(axes[1, 1], df, "instruction_tokens",
                     "program size (tokens)")
    _violin_by_shard(axes[1, 2], df, "instruction_chars",
                     "program size (chars)")
    _scatter_with_fit(axes[2, 0], df, "instruction_tokens",
                      "optimized_score", "quality vs tokens")
    _scatter_with_fit(axes[2, 1], df, "instruction_tokens", "delta",
                      "lift vs tokens")
    _scatter_with_fit(axes[2, 2], df, "duration_s", "optimized_score",
                      "quality vs duration")
    fig.suptitle(f"GEPA MovieLens analysis  (n={len(df)} cohorts)",
                 fontsize=16)
    _save(fig, out_dir / "overview.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shards", nargs="+", required=True,
        help="One or more shard directories (each containing result_*.json "
             "and program_*.json)")
    p.add_argument(
        "--out", required=True,
        help="Output directory for summary.csv and plots")
    p.add_argument(
        "--repo-root", default=None,
        help="Repo root used to resolve optimized_program_path. Defaults to "
             "the current working directory.")
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot rendering (only dump CSV + summary)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    shard_paths = [Path(s) for s in args.shards]
    for s in shard_paths:
        if not s.is_dir():
            raise SystemExit(f"Not a directory: {s}")

    df = build_dataframe(shard_paths, repo_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary.csv"
    df.sort_values(["shard", "sample_idx"]).to_csv(csv_path, index=False)
    print(f"[write] {csv_path}  ({len(df)} rows)")

    stats_path = out_dir / "stats.csv"
    summarize(df).to_csv(stats_path)
    print(f"[write] {stats_path}")

    print_summary(df)

    if not args.no_plots:
        render_plots(df, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
