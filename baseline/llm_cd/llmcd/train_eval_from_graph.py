from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.llm_cd_baseline import evaluate_parent_predictor  # noqa: E402
from baseline.llmcd.common import load_config, load_configured_data, read_json, task_type_from_v, write_json  # noqa: E402


def run(config_path: str, graph_path: str, run_dir: str):
    config = load_config(config_path)
    graph_payload = read_json(graph_path)
    data, y, v, num_classes_dict, target_idx, feature_names, descriptions = load_configured_data(config)
    feature_names = graph_payload.get("feature_names", feature_names)
    graph = np.asarray(graph_payload["graph"], dtype=int)
    parents = graph_payload.get("parents")
    if parents is None:
        parents = sorted(np.where(graph[:, target_idx] == 1)[0].astype(int).tolist())
    parents = [int(idx) for idx in parents]

    train_cfg = config.get("training", {})
    predictor = train_cfg.get("predictor", "rf")
    random_state = int(config.get("random_state", 42))
    task_type = graph_payload.get("task_type") or task_type_from_v(v, target_idx)
    metrics = evaluate_parent_predictor(
        data,
        y,
        target_idx=target_idx,
        parents=parents,
        task_type=task_type,
        predictor=predictor,
        random_state=random_state,
    )
    payload = {
        "stage": "train_eval",
        "config_path": str(config_path),
        "graph_path": str(graph_path),
        "dataset": config["dataset"],
        "task_type": task_type,
        "target_idx": int(target_idx),
        "target_name": feature_names[target_idx],
        "parents": parents,
        "parent_names": [feature_names[idx] for idx in parents],
        "predictor": predictor,
        "metrics": metrics,
    }
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    write_json(str(out_path), payload)
    print(out_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: train and evaluate from a final graph")
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run(args.config, args.graph, args.run_dir)


if __name__ == "__main__":
    main()
