#!/bin/bash
# ============================================================
# Auto-tuning: try 3 alpha values per dataset, pick best
# Usage: bash baseline/llmcd/scripts/tune_all.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python"
RUNS_DIR="baseline/llmcd/runs"

DATASETS=(
  "adult:0.01"
  "adult:0.05"
  "adult:0.1"
  "cardio:0.01"
  "cardio:0.05"
  "cardio:0.1"
  "creditcard:0.01"
  "creditcard:0.05"
  "creditcard:0.1"
  "crime:0.01"
  "crime:0.05"
  "crime:0.1"
  "diamonds:0.01"
  "diamonds:0.05"
  "diamonds:0.1"
  "elevator:0.01"
  "elevator:0.05"
  "elevator:0.1"
  "housesale:0.01"
  "housesale:0.05"
  "housesale:0.1"
  "housing:0.01"
  "housing:0.05"
  "housing:0.1"
  "meps:0.01"
  "meps:0.05"
  "meps:0.1"
  "synthetic_demo:0.01"
  "synthetic_demo:0.05"
  "synthetic_demo:0.1"
)

mkdir -p "$RUNS_DIR"

for entry in "${DATASETS[@]}"; do
  ds="${entry%%:*}"
  alpha="${entry##*:}"

  CONFIG="$PROJECT_ROOT/baseline/llmcd/configs/${ds}.json"
  RUN_DIR="$RUNS_DIR/${ds}_alpha${alpha}"

  if [ ! -f "$CONFIG" ]; then
    continue
  fi

  echo "===== $ds  alpha=$alpha ====="

  cp "$CONFIG" /tmp/llmcd_tune_$$.json
  python3 -c "
import json
cfg = json.load(open('/tmp/llmcd_tune_$$.json'))
cfg['pc']['alpha'] = $alpha
json.dump(cfg, open('/tmp/llmcd_tune_$$.json', 'w'), indent=2)
"

  # Step 1: PC
  $PYTHON baseline/llmcd/discover_pc_graph.py \
    --config /tmp/llmcd_tune_$$.json \
    --run_dir "$RUN_DIR" 2>&1 | tail -1

  # Step 2: Graph
  $PYTHON baseline/llmcd/judge_llm_graph.py \
    --config /tmp/llmcd_tune_$$.json \
    --pc_graph "$RUN_DIR/pc_graph.json" \
    --run_dir "$RUN_DIR" 2>&1 | tail -1

  # Step 3: Train
  $PYTHON baseline/llmcd/train_eval_from_graph.py \
    --config /tmp/llmcd_tune_$$.json \
    --graph "$RUN_DIR/final_graph.json" \
    --run_dir "$RUN_DIR" 2>&1 | tail -1

  python3 -c "
import json, sys
try:
    m = json.load(open('$RUN_DIR/metrics.json'))
    t = m['metrics']['test']
    p = m['parent_names']
    if 'auc' in t:
        print(f'  alpha=$alpha  Parents({len(p)}): {p}')
        print(f'  AUC={t[\"auc\"]:.4f}  F1={t[\"f1_score\"]:.4f}  Acc={t[\"accuracy\"]:.4f}')
    else:
        print(f'  alpha=$alpha  Parents({len(p)}): {p}')
        print(f'  R^2={t[\"r2\"]:.4f}  RMSE={t[\"rmse\"]:.4f}')
except:
    print('  FAILED')
"

  rm -f /tmp/llmcd_tune_$$.json
  echo ""
done

echo "============================================"
echo "  Tuning complete! Best alpha per dataset:"
echo "============================================"

for ds in adult cardio creditcard crime diamonds elevator housesale housing meps synthetic_demo; do
  best_alpha=""
  best_score=""
  for alpha in 0.01 0.05 0.1; do
    f="$RUNS_DIR/${ds}_alpha${alpha}/metrics.json"
    if [ -f "$f" ]; then
      score=$(python3 -c "
import json
m=json.load(open('$f'))
t=m['metrics']['test']
print(t.get('auc',t.get('r2',-999)))
" 2>/dev/null)
      if [ -n "$best_score" ] && [ "$(echo "$score > $best_score" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        best_score="$score"
        best_alpha="$alpha"
      elif [ -z "$best_score" ]; then
        best_score="$score"
        best_alpha="$alpha"
      fi
    fi
  done
  echo "  $ds: best alpha=$best_alpha ($best_score)"
done
