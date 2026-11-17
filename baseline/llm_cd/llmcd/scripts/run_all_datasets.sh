#!/bin/bash
# ============================================================
# Run LLM-CD baseline for all datasets (3 steps: PC -> Graph -> Train)
# Usage:
#   chmod +x baseline/llmcd/scripts/run_all_datasets.sh
#   bash baseline/llmcd/scripts/run_all_datasets.sh
#
# For GPU clusters use the Slurm version (see run_all_slurm.sh)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Config ----
PYTHON=".venv/bin/python"
RUNS_DIR="baseline/llmcd/runs"
CONFIG_DIR="baseline/llmcd/configs"

DATASETS=(
  "synthetic"
  "adult"
  "cardio"
  "creditcard"
  "crime"
  "diamonds"
  "elevator"
  "housesale"
  "meps"
)

mkdir -p "$RUNS_DIR"

for ds in "${DATASETS[@]}"; do
  CONFIG="$CONFIG_DIR/${ds}.json"
  RUN_DIR="$RUNS_DIR/${ds}_llmcd"

  if [ ! -f "$CONFIG" ]; then
    echo "[SKIP] $ds — config file $CONFIG not found"
    continue
  fi

  echo "============================================"
  echo "  $ds"
  echo "============================================"

  # ---- Stage 1: PC causal discovery ----
  echo "  [1/3] PC discovery..."
  $PYTHON baseline/llmcd/discover_pc_graph.py \
    --config "$CONFIG" \
    --run_dir "$RUN_DIR"

  # ---- Stage 2: Graph judgment (no LLM, fallback rules) ----
  echo "  [2/3] Graph judgment (fallback)..."
  $PYTHON baseline/llmcd/judge_llm_graph.py \
    --config "$CONFIG" \
    --pc_graph "$RUN_DIR/pc_graph.json" \
    --run_dir "$RUN_DIR"

  # ---- Stage 3: Train & eval using parent nodes ----
  echo "  [3/3] Train & eval..."
  $PYTHON baseline/llmcd/train_eval_from_graph.py \
    --config "$CONFIG" \
    --graph "$RUN_DIR/final_graph.json" \
    --run_dir "$RUN_DIR"

  echo "  Done -> $RUN_DIR/metrics.json"
  echo ""
done

echo "============================================"
echo "  All done! Summary:"
echo "============================================"
for ds in "${DATASETS[@]}"; do
  RUN_DIR="$RUNS_DIR/${ds}_llmcd"
  if [ -f "$RUN_DIR/metrics.json" ]; then
    echo "  $ds OK"
  else
    echo "  $ds FAIL (no metrics)"
  fi
done
