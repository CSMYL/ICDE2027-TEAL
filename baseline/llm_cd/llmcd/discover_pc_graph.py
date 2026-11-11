from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.llm_cd_baseline import _choose_independence_test  # noqa: E402
from baseline.llmcd.common import load_config, load_configured_data, sample_for_pc, task_type_from_v, write_json  # noqa: E402


def run(config_path: str, run_dir: str):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT

    config = load_config(config_path)
    data, y, v, num_classes_dict, target_idx, feature_names, descriptions = load_configured_data(config)
    pc_cfg = config.get("pc", {})
    random_state = int(config.get("random_state", 42))
    sample_size = pc_cfg.get("sample_size", config.get("sample_size", 1000))
    pc_data, sample_indices = sample_for_pc(data, sample_size, random_state)
    indep_test = _choose_independence_test(v, pc_cfg.get("independence_test", "auto"))
    alpha = float(pc_cfg.get("alpha", 0.05))

    try:
        cg = pc(pc_data, alpha=alpha, indep_test=indep_test, stable=True, uc_rule=0, uc_priority=2, show_progress=False)
        pc_input = pc_data
        jitter_used = False
    except ValueError as exc:
        if "singular" not in str(exc).lower() or indep_test not in {"fisherz", "kci"}:
            raise
        rng = np.random.default_rng(random_state)
        pc_input = pc_data + rng.normal(0.0, float(pc_cfg.get("jitter_scale", 1e-6)), size=pc_data.shape)
        cg = pc(pc_input, alpha=alpha, indep_test=indep_test, stable=True, uc_rule=0, uc_priority=2, show_progress=False)
        jitter_used = True

    directed_edges = [(int(i), int(j)) for i, j in cg.find_fully_directed()]
    undirected_edges = sorted({tuple(sorted((int(i), int(j)))) for i, j in cg.find_undirected()})

    uncertain = []
    threshold = float(pc_cfg.get("ci_threshold", 0.001))
    max_pairs = int(pc_cfg.get("max_uncertain_pairs", 100))
    if threshold > 0 and max_pairs > 0:
        cit = CIT(pc_input, indep_test)
        candidates = []
        for i in range(pc_input.shape[1]):
            for j in range(i + 1, pc_input.shape[1]):
                try:
                    p_value = float(cit(i, j, ()))
                except Exception:
                    continue
                delta = abs(p_value - alpha)
                if delta <= threshold:
                    candidates.append((delta, i, j, p_value))
        candidates.sort(key=lambda item: item[0])
        for delta, i, j, p_value in candidates[:max_pairs]:
            uncertain.append({
                "x": int(i),
                "y": int(j),
                "conditioning_set": [],
                "p_value": float(p_value),
                "delta_from_alpha": float(delta),
            })

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "pc_discovery",
        "config_path": str(config_path),
        "dataset": config["dataset"],
        "task_type": task_type_from_v(v, target_idx),
        "target_idx": int(target_idx),
        "target_name": feature_names[target_idx],
        "feature_names": feature_names,
        "feature_descriptions": descriptions,
        "v": v.astype(int).tolist(),
        "num_classes_dict": {str(k): int(val) for k, val in num_classes_dict.items()},
        "alpha": alpha,
        "independence_test": indep_test,
        "sample_size": int(pc_data.shape[0]),
        "sample_indices": sample_indices,
        "jitter_used": jitter_used,
        "pc_directed_edges": directed_edges,
        "pc_undirected_edges": undirected_edges,
        "uncertain_ci_pairs": uncertain,
    }
    out_path = output_dir / "pc_graph.json"
    write_json(str(out_path), payload)
    print(out_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: run PC and write pc_graph.json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run(args.config, args.run_dir)


if __name__ == "__main__":
    main()
