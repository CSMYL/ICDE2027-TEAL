#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-baseline/llmcd/configs/synthetic_demo.json}"
USE_LLM="${USE_LLM:-0}"

RUN_NAME="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name","llmcd_run"))' "$CONFIG")"
RUN_DIR="baseline/llmcd/runs/${RUN_NAME}"
mkdir -p "$RUN_DIR" .cache/matplotlib

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"

.venv/bin/python baseline/llmcd/discover_pc_graph.py \
  --config "$CONFIG" \
  --run_dir "$RUN_DIR"

if [[ "$USE_LLM" == "1" ]]; then
  .venv/bin/python baseline/llmcd/judge_llm_graph.py \
    --config "$CONFIG" \
    --pc_graph "$RUN_DIR/pc_graph.json" \
    --run_dir "$RUN_DIR" \
    --use_llm
else
  .venv/bin/python baseline/llmcd/judge_llm_graph.py \
    --config "$CONFIG" \
    --pc_graph "$RUN_DIR/pc_graph.json" \
    --run_dir "$RUN_DIR"
fi

.venv/bin/python baseline/llmcd/train_eval_from_graph.py \
  --config "$CONFIG" \
  --graph "$RUN_DIR/final_graph.json" \
  --run_dir "$RUN_DIR"

echo "LLM-CD run finished: $RUN_DIR"
