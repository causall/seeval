#!/usr/bin/env bash
# Run GEPA on cohorts 1..10 of the seed=48 / 100m source, capped at 140 examples
# per cohort, fanned out 10-wide through the local LiteLLM proxy.
#
# Usage:
#   bash scripts/run_s48_n140.sh              # full pipeline (produce + run)
#   bash scripts/run_s48_n140.sh produce      # just (re)build shards
#   bash scripts/run_s48_n140.sh run          # just run GEPA over existing shards
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="src/experiment/movielens/rust_output_100m_config.json"
OUT_DIR="shards/fixed_s48_n140_glm_5"
START=150          # inclusive
END=151           # exclusive  (so indices 1..10 = 10 cohorts)
TOTAL_EXAMPLES=140
WORKERS=10

MODEL="openai/glm-5"
API_BASE="http://localhost:4000"
API_KEY="noop"

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

step="${1:-all}"

produce() {
  echo "[produce] start=$START end=$END total=$TOTAL_EXAMPLES -> $OUT_DIR"
  python -m experiment.movielens.distributed_main produce \
    --config "$CONFIG" \
    --out "$OUT_DIR" \
    --start "$START" \
    --end "$END" \
    --total-examples "$TOTAL_EXAMPLES" \
    2>&1 | tee "$LOG_DIR/produce.log"
}

run_one() {
  local i="$1"
  local result="$OUT_DIR/result_${i}.json"
  if [[ -f "$result" ]]; then
    echo "[run] idx=$i already done ($result), skipping"
    return 0
  fi
  python -m experiment.movielens.distributed_main run \
    --dataset-dir "$OUT_DIR" \
    --sample-idx "$i" \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --api-key "$API_KEY" \
    --out "$result" \
    --save-program "$OUT_DIR/program_${i}.json" \
    >"$LOG_DIR/run_${i}.log" 2>&1
}
export -f run_one
export OUT_DIR MODEL API_BASE API_KEY LOG_DIR

run_all() {
  echo "[run] cohorts $START..$((END-1)) with $WORKERS workers"
  local t0=$(date +%s)
  seq "$START" $((END-1)) \
    | xargs -n1 -P"$WORKERS" -I{} bash -c 'run_one "$1"' _ {}
  local t1=$(date +%s)
  echo "[run] done in $((t1 - t0))s"
  echo "[run] summary:"
  for i in $(seq "$START" $((END-1))); do
    local f="$OUT_DIR/result_${i}.json"
    if [[ -f "$f" ]]; then
      python -c "
import json, sys
r = json.load(open('$f'))
print(f\"  idx={r['sample_idx']:>3}  baseline={r['baseline_score']:>6.2f}  optimized={r['optimized_score']:>6.2f}  dur={r['duration_s']:>6.1f}s  train={r['n_train']} val={r['n_val']} test={r['n_test']}\")
"
    else
      echo "  idx=$i  MISSING"
    fi
  done
}

case "$step" in
  produce) produce ;;
  run)     run_all ;;
  all)     produce && run_all ;;
  *) echo "unknown step: $step (use: produce|run|all)"; exit 1 ;;
esac
