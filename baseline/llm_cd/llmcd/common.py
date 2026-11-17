from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from baseline.llm_cd_baseline import feature_metadata, load_baseline_arrays


def read_json(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def write_json(path: str, payload: Dict):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_config(config_path: str) -> Dict:
    config = read_json(config_path)
    config.setdefault("dataset", "synthetic")
    config.setdefault("run_name", config["dataset"])
    config.setdefault("pc", {})
    config.setdefault("llm", {})
    config.setdefault("training", {})
    return config


def load_configured_data(config: Dict):
    data_cfg = config.get("data", {})
    if data_cfg.get("table_path"):
        return _load_table_data(config)

    dataset = config["dataset"]
    full, y, v, num_classes_dict, target_idx = load_baseline_arrays(dataset)
    feature_names, descriptions = feature_metadata(dataset, full.shape[1])
    feature_names, descriptions = apply_feature_overrides(config, feature_names, descriptions)
    if data_cfg.get("target_idx") is not None:
        target_idx = int(data_cfg["target_idx"])
        y = full[:, target_idx]
    return full, y, v, num_classes_dict, target_idx, feature_names, descriptions


def apply_feature_overrides(config: Dict, feature_names: List[str], descriptions: Dict[str, str]):
    features = config.get("features") or []
    if not features:
        return feature_names, descriptions

    names = feature_names[:]
    desc = descriptions.copy()
    for item in features:
        idx = int(item["index"])
        while len(names) <= idx:
            names.append(f"feature_{len(names)}")
        name = str(item.get("name") or names[idx])
        names[idx] = name
        desc[name] = str(item.get("description") or name)
    return names, {name: desc.get(name, name) for name in names}


def _load_table_data(config: Dict):
    data_cfg = config["data"]
    table_path = Path(data_cfg["table_path"])
    header = data_cfg.get("header", "infer")
    header_arg = 0 if header == "infer" else None if header is None else int(header)
    df = pd.read_csv(table_path, header=header_arg)
    if data_cfg.get("drop_columns"):
        df = df.drop(columns=data_cfg["drop_columns"])

    target_col = data_cfg.get("target_column")
    target_idx = len(df.columns) - 1
    if data_cfg.get("target_idx") is not None:
        target_idx = int(data_cfg["target_idx"])
    if target_col is not None:
        target_idx = list(df.columns).index(target_col)

    feature_names = [str(c) for c in df.columns]
    if header_arg is None:
        feature_names = [f"feature_{i}" for i in range(df.shape[1])]

    categorical_indices = set(map(int, data_cfg.get("categorical_indices", [])))
    continuous_indices = set(map(int, data_cfg.get("continuous_indices", [])))
    if not categorical_indices and not continuous_indices:
        for i, col in enumerate(df.columns):
            if df[col].dtype == object:
                categorical_indices.add(i)
            else:
                continuous_indices.add(i)
    for idx in range(df.shape[1]):
        if idx not in categorical_indices and idx not in continuous_indices:
            continuous_indices.add(idx)

    work_df = df.copy()
    num_classes_dict = {}
    for idx in categorical_indices:
        encoder = LabelEncoder()
        work_df.iloc[:, idx] = encoder.fit_transform(work_df.iloc[:, idx].astype(str))
        num_classes_dict[idx] = len(encoder.classes_)
    if data_cfg.get("standardize_continuous", True):
        cont = sorted(continuous_indices)
        if cont:
            scaler = StandardScaler()
            work_df.iloc[:, cont] = scaler.fit_transform(work_df.iloc[:, cont])

    full = work_df.to_numpy(dtype=float)
    y = full[:, target_idx]
    v = np.zeros(df.shape[1], dtype=int)
    for idx in categorical_indices:
        v[idx] = 1

    descriptions = {name: f"Table column {name}." for name in feature_names}
    feature_names, descriptions = apply_feature_overrides(config, feature_names, descriptions)
    return full, y, v, num_classes_dict, target_idx, feature_names, descriptions


def sample_for_pc(data: np.ndarray, sample_size: Optional[int], random_state: int) -> Tuple[np.ndarray, List[int]]:
    if sample_size is None or sample_size <= 0 or data.shape[0] <= sample_size:
        indices = np.arange(data.shape[0])
        return data, indices.astype(int).tolist()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(data.shape[0], size=sample_size, replace=False)
    return data[indices], indices.astype(int).tolist()


def task_type_from_v(v: np.ndarray, target_idx: int) -> str:
    return "regression" if int(v[target_idx]) == 0 else "classification"
