# Rust MovieLens Setup Generator

`movielens_setup_rust` — fast, isolated Rust binary that generates the
**cohort manifest** (`*.jsonl`) consumed by the rest of the MovieLens
GEPA pipeline (`main.py`, `distributed_main.py`, `shards.py`).

## Why this exists

A "cohort" is a small group of `N` MovieLens users whose **commonly
rated movies** (set intersection of each user's rated movies) is large
enough to be experimentally useful — i.e. ≥ `valid_movie_count` shared
titles. Each such cohort becomes one experimental unit downstream
(train/val/test split is drawn from the intersection).

Enumerating every candidate cohort is combinatorially infeasible:

- After filtering MovieLens-32M to "popular" movies (≥ `MIN_REVIEWS=1000`
  ratings) and "active" users (≥ `MIN_INDIVIDUAL_USER_REVIEWS=200`
  ratings on those movies), you still get on the order of 10⁴ users.
- For `num_users = 5`, the search space is `C(~10⁴, 5) ≈ 10¹⁶` — you
  can't enumerate, sort, or score those, and the vast majority have
  near-empty intersections anyway.
- The pure-Python equivalent in `group_dataset.py` (which uses
  `itertools.combinations` + pandas) does not scale past a few thousand
  candidates.

So this binary does **Monte Carlo rejection sampling** instead: draw
`num_runs` random `num_users`-tuples, intersect their sorted rated-movie
vectors (linear two-pointer merge), keep only the tuples whose
intersection size exceeds `valid_movie_count`, sort the survivors by
intersection size, and write them out.

Rewriting it in Rust drops a full pipeline cold-start from minutes
(Python) to seconds (Rust), and the bincode cache makes warm runs
near-instant.

## Output schema

Each JSONL line:

```json
{"movie_ids":[1,32,...],"user_ids":[123,456,...],"valid_movie_count":137}
```

`movie_ids` is the sorted intersection of every selected user's rated
movies. `user_ids` is sorted ascending. Downstream code keys cohorts by
position in this file (cohort index 1..N).

## Run

From repo root:

```bash
cargo run --manifest-path src/experiment/movielens/rust_setup/Cargo.toml -- \
  --setup src/experiment/movielens/rust_output.jsonl \
  --num-users 5 \
  --num-runs 10000 \
  --valid-movie-count 50 \
  --seed 43
```

Writes:
- `src/experiment/movielens/rust_output.jsonl`
- `src/experiment/movielens/rust_output_config.json`

## Fast smoke run

```bash
cargo run --manifest-path src/experiment/movielens/rust_setup/Cargo.toml -- \
  --setup src/experiment/movielens/rust_smoke.jsonl \
  --num-runs 20 \
  --valid-movie-count 0 \
  --seed 45
```

`--valid-movie-count 0` keeps every sampled cohort (intersection just
has to be non-empty) — handy for plumbing tests.

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--setup` | _required_ | Output JSONL path; config is written next to it as `<stem>_config.json`. |
| `--num-users` | `5` | Cohort size `N`. Must be `≤` filtered user count. |
| `--num-runs` | `10000` | Monte Carlo trials. Survivors ≤ this. |
| `--valid-movie-count` | `50` | Strict-greater-than threshold for keeping a cohort. |
| `--seed` | `43` | `StdRng` seed; fully reproducible output for fixed inputs. |

## Cache

Binary cache:

`.dataset/.cache/indices_ml-32m_minrev1000_minind200_rust.bin`

Cold run builds the cache; subsequent runs reuse it as long as
`movies.csv` and `ratings.csv` `(length, mtime)` signatures are
unchanged and the two threshold constants
(`MIN_REVIEWS`, `MIN_INDIVIDUAL_USER_REVIEWS`) match the blob. Bumping
either constant in `main.rs` forces a rebuild.

## Isolation / boundaries

This crate is intentionally standalone:

- No workspace `Cargo.toml`; build it with `--manifest-path` only.
- It reads `.dataset/ml-32m/*.csv` and writes JSONL + a sibling config
  JSON. Nothing else is touched.
- It does not import or know about any Python code; the Python pipeline
  consumes its JSONL via `group_dataset.load_sample_results_from_disk`
  (the schema matches `SystematicSampleResult` / `SetupConfig` 1:1).
