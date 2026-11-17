from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.llm_cd_baseline import LLMJudge, _break_cycles, _fallback_target_parents  # noqa: E402
from baseline.llm_cd_prompts import conditional_independence_prompt, directed_edge_prompt, undirected_edge_prompt  # noqa: E402
from baseline.llmcd.common import load_config, load_configured_data, read_json, write_json  # noqa: E402


def run(config_path: str, pc_graph_path: str, run_dir: str, use_llm: bool):
    config = load_config(config_path)
    pc_payload = read_json(pc_graph_path)
    data, y, v, num_classes_dict, target_idx, feature_names, descriptions = load_configured_data(config)
    feature_names = pc_payload.get("feature_names", feature_names)
    descriptions = pc_payload.get("feature_descriptions", descriptions)
    n_features = len(feature_names)
    graph = np.zeros((n_features, n_features), dtype=int)

    directed_edges = [tuple(map(int, edge)) for edge in pc_payload.get("pc_directed_edges", [])]
    undirected_edges = [tuple(map(int, edge)) for edge in pc_payload.get("pc_undirected_edges", [])]
    llm_cfg = config.get("llm", {})
    cache_path = llm_cfg.get("cache_path") or str(Path(run_dir) / "llm_cache.json")
    llm = LLMJudge(
        enabled=use_llm,
        api_key=None,
        base_url=llm_cfg.get("base_url"),
        model=llm_cfg.get("model"),
        cache_path=cache_path,
        sleep_seconds=float(llm_cfg.get("sleep_seconds", 0.0)),
    )

    ci_decisions = []
    for item in pc_payload.get("uncertain_ci_pairs", []):
        x = int(item["x"])
        y_idx = int(item["y"])
        cond = [int(i) for i in item.get("conditioning_set", [])]
        score = None
        confidence = None
        if use_llm:
            prompt = conditional_independence_prompt(x, y_idx, cond, feature_names, descriptions)
            score, confidence = llm.score_independence(prompt)
            has_directed = any(set(edge) == {x, y_idx} for edge in directed_edges)
            has_undirected = any(set(edge) == {x, y_idx} for edge in undirected_edges)
            if score >= 0.5:
                directed_edges = [edge for edge in directed_edges if set(edge) != {x, y_idx}]
                undirected_edges = [edge for edge in undirected_edges if set(edge) != {x, y_idx}]
            elif not has_directed and not has_undirected:
                undirected_edges.append(tuple(sorted((x, y_idx))))
        ci_decisions.append({**item, "llm_score": score, "llm_confidence": confidence})

    edge_decisions = []
    for i, j in directed_edges:
        action = "KEEP"
        confidence = None
        if use_llm:
            prompt = directed_edge_prompt(i, j, feature_names, descriptions)
            action, confidence = llm.score_directed_edge(prompt)
        if action == "KEEP":
            graph[i, j] = 1
        elif action == "FLIP":
            graph[j, i] = 1
        edge_decisions.append({"source": i, "target": j, "type": "directed", "action": action, "confidence": confidence})

    for i, j in sorted(set(tuple(sorted(edge)) for edge in undirected_edges)):
        direction = 1
        confidence = None
        if use_llm:
            prompt = undirected_edge_prompt(i, j, feature_names, descriptions)
            direction, confidence = llm.score_undirected_edge(prompt)
        if direction == 1:
            graph[i, j] = 1
            source, target = i, j
        else:
            graph[j, i] = 1
            source, target = j, i
        edge_decisions.append({
            "source": source,
            "target": target,
            "type": "undirected",
            "action": "ORIENT",
            "confidence": confidence,
        })

    graph = _break_cycles(graph, data, feature_names, descriptions, llm if use_llm else None)
    parents = sorted(np.where(graph[:, target_idx] == 1)[0].astype(int).tolist())
    if not parents:
        parents = _fallback_target_parents(data, target_idx, k=min(5, max(1, n_features - 1)))
        for parent in parents:
            graph[parent, target_idx] = 1

    payload = {
        "stage": "llm_graph_judgment" if use_llm else "deterministic_graph_judgment",
        "config_path": str(config_path),
        "pc_graph_path": str(pc_graph_path),
        "dataset": pc_payload["dataset"],
        "task_type": pc_payload["task_type"],
        "target_idx": int(target_idx),
        "target_name": feature_names[target_idx],
        "feature_names": feature_names,
        "feature_descriptions": descriptions,
        "graph": graph.astype(int).tolist(),
        "parents": parents,
        "parent_names": [feature_names[idx] for idx in parents],
        "ci_decisions": ci_decisions,
        "edge_decisions": edge_decisions,
        "llm_enabled": use_llm,
        "llm_calls": llm.calls,
        "llm_cache_hits": llm.cache_hits,
    }
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "final_graph.json"
    write_json(str(out_path), payload)
    print(out_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: use LLM to refine PC graph and write final_graph.json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pc_graph", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--use_llm", action="store_true")
    args = parser.parse_args()
    run(args.config, args.pc_graph, args.run_dir, args.use_llm)


if __name__ == "__main__":
    main()
