#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
_RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, 'raw_data')

DATASETS = {
    'crime': {
        'source': os.path.join(_RAW_DATA_DIR, 'crime.csv'),
        'target_dir': os.path.dirname(os.path.abspath(__file__)),
        'task': 'regression',
        'continuous_count': 119,
        'categorical_count': 3
    },
    'meps': {
        'source': os.path.join(_RAW_DATA_DIR, 'meps.csv'),
        'target_dir': os.path.dirname(os.path.abspath(__file__)),
        'task': 'regression',
        'continuous_features': ['AGE', 'PCS42', 'MCS42', 'K6SUM42']
    }
}

def prepare_tabgnn_dataset(dataset_name, config):
    """Prepare dataset for TabGNN."""
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Source: {config['source']}")

    df = pd.read_csv(config['source'], header=0)

    if dataset_name == 'meps':
        cols_to_drop = [c for c in df.columns if 'Unnamed' in c or 'PERWT' in c]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns: {cols_to_drop}")

    ds_info = {
        'task': config['task'],
        'columns': []
    }

    if dataset_name == 'crime':
        feature_cols = df.columns[:-1]
        for i, col in enumerate(feature_cols):
            if i < config['continuous_count']:
                ds_info['columns'].append({'name': f'feature{i}', 'type': 'NUMERIC'})
            else:
                unique_vals = int(df[col].nunique())
                ds_info['columns'].append({'name': f'feature{i}', 'type': 'CATEGORICAL', 'cardinality': unique_vals})
    elif dataset_name == 'meps':
        feature_cols = df.columns[:-1]
        feature_idx = 0
        for col in feature_cols:
            if col in config['continuous_features']:
                ds_info['columns'].append({'name': f'feature{feature_idx}', 'type': 'NUMERIC'})
            else:
                unique_vals = int(df[col].nunique())
                ds_info['columns'].append({'name': f'feature{feature_idx}', 'type': 'CATEGORICAL', 'cardinality': unique_vals})
            feature_idx += 1

    ds_info['columns'].append({'name': 'TARGET', 'type': 'NUMERIC'})

    ds_info_path = os.path.join(config['target_dir'], 'data', 'test_data', f'{dataset_name}.ds_info.json')
    os.makedirs(os.path.dirname(ds_info_path), exist_ok=True)
    with open(ds_info_path, 'w', encoding='utf-8') as f:
        json.dump(ds_info, f, indent=2, ensure_ascii=False)
    print(f"ds_info.json saved to: {ds_info_path}")

    raw_data_dir = os.path.join(config['target_dir'], 'data', 'raw_data')
    os.makedirs(raw_data_dir, exist_ok=True)

    df_renamed = df.copy()
    new_columns = []
    for col_info in ds_info['columns']:
        if col_info['name'] == 'TARGET':
            original_label_col = df.columns[-1]
            new_columns.append(original_label_col)
        else:
            col_idx = int(col_info['name'].replace('feature', ''))
            if dataset_name == 'crime':
                original_col = df.columns[col_idx]
            else:
                feature_cols = [c for c in df.columns if c not in cols_to_drop and c != df.columns[-1]]
                original_col = feature_cols[col_idx]
            new_columns.append(original_col)

    df_renamed = df_renamed[new_columns]

    rename_dict = {}
    for i, col_info in enumerate(ds_info['columns']):
        rename_dict[new_columns[i]] = col_info['name']
    df_renamed = df_renamed.rename(columns=rename_dict)

    csv_path = os.path.join(raw_data_dir, f'{dataset_name}.csv')
    df_renamed.to_csv(csv_path, index=False, header=False)
    print(f"Data saved to: {csv_path}")
    print(f"Data shape: {df_renamed.shape}")

    return ds_info, df_renamed.shape

if __name__ == '__main__':
    print("=" * 60)
    print("Preparing datasets for TabGNN")
    print("=" * 60)

    results = {}
    for dataset_name, config in DATASETS.items():
        if not os.path.exists(config['source']):
            print(f"\nWarning: source file not found: {config['source']}")
            continue
        try:
            ds_info, shape = prepare_tabgnn_dataset(dataset_name, config)
            results[dataset_name] = {
                'shape': shape,
                'num_features': len(ds_info['columns']) - 1,
                'task': ds_info['task']
            }
        except Exception as e:
            print(f"\nError processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    for dataset_name, info in results.items():
        print(f"{dataset_name:12s}: {info['shape'][0]:6d} rows, {info['num_features']:3d} features, task: {info['task']}")
    print("=" * 60)
