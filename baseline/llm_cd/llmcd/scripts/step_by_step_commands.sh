#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-baseline/llmcd/configs/synthetic_demo.json}"
RUN_NAME="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name","llmcd_run"))' "$CONFIG")"
RUN_DIR="baseline/llmcd/runs/${RUN_NAME}"

cat <<EOF
################################################################################
# LLM-CD Step-by-Step Commands
################################################################################
# Config: $CONFIG
# Run directory: $RUN_DIR
#
# Output artifacts:
#   1. pc_graph.json     : Step 1 output, initial graph from PC algorithm
#   2. final_graph.json  : Step 2 output, final graph after LLM or fallback rules
#   3. metrics.json      : Step 3 output, train/test results using target parent nodes
################################################################################

################################################################################
# 0. Optional: if running on a machine with internet access to call LLM APIs,
#    set these environment variables first.
#    NOTE: never write real API keys into code or config files.
################################################################################
export LLMCD_API_KEY="replace-with-your-api-key"
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"
export MPLCONFIGDIR="\$PWD/.cache/matplotlib"
mkdir -p "$RUN_DIR" .cache/matplotlib

################################################################################
# 1. Offline-capable step: input data -> PC initial graph
#
# Input:
#   - $CONFIG
#   - Data specified in config, or project built-in CAIM loader
#
# Output:
#   - $RUN_DIR/pc_graph.json
#
# After this step, if the server has no internet, copy the following files to
# a machine with internet access:
#   - $CONFIG
#   - $RUN_DIR/pc_graph.json
################################################################################
.venv/bin/python baseline/llmcd/discover_pc_graph.py \\
  --config "$CONFIG" \\
  --run_dir "$RUN_DIR"

################################################################################
# 2A. Internet-required step: PC initial graph -> LLM-judged final graph
#
# Prerequisites:
#   - LLMCD_API_KEY / LLMCD_BASE_URL / LLMCD_MODEL must be set
#
# Input:
#   - $CONFIG
#   - $RUN_DIR/pc_graph.json
#
# Output:
#   - $RUN_DIR/final_graph.json
#
# After this step, copy final_graph.json back to the offline server to train.
################################################################################
.venv/bin/python baseline/llmcd/judge_llm_graph.py \\
  --config "$CONFIG" \\
  --pc_graph "$RUN_DIR/pc_graph.json" \\
  --run_dir "$RUN_DIR" \\
  --use_llm

################################################################################
# 2B. Offline fallback: PC initial graph -> rule-oriented final graph, no API
#
# Use this instead of 2A if no API is available, or for quick pipeline testing.
#
# Input:
#   - $CONFIG
#   - $RUN_DIR/pc_graph.json
#
# Output:
#   - $RUN_DIR/final_graph.json
################################################################################
.venv/bin/python baseline/llmcd/judge_llm_graph.py \\
  --config "$CONFIG" \\
  --pc_graph "$RUN_DIR/pc_graph.json" \\
  --run_dir "$RUN_DIR"

################################################################################
# 3. Offline-capable step: final graph -> train and test on target parent nodes
#
# Input:
#   - $CONFIG
#   - $RUN_DIR/final_graph.json
#
# Output:
#   - $RUN_DIR/metrics.json
################################################################################
.venv/bin/python baseline/llmcd/train_eval_from_graph.py \\
  --config "$CONFIG" \\
  --graph "$RUN_DIR/final_graph.json" \\
  --run_dir "$RUN_DIR"
EOF
