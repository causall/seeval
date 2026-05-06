# Rust MovieLens Setup Generator

Tiny usage guide for `movielens_setup_rust`.

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

This writes:
- `src/experiment/movielens/rust_output.jsonl`
- `src/experiment/movielens/rust_output_config.json`

## Fast Smoke Run

```bash
cargo run --manifest-path src/experiment/movielens/rust_setup/Cargo.toml -- \
  --setup src/experiment/movielens/rust_smoke.jsonl \
  --num-runs 20 \
  --valid-movie-count 0 \
  --seed 45
```

## Cache

The binary cache is written at:

`.dataset/.cache/indices_ml-32m_minrev1000_minind200_rust.bin`

Cold run builds cache; later runs reuse it when `movies.csv` and `ratings.csv` signatures are unchanged.
