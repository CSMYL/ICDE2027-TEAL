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
        'ds_info_path': 'data/test_data/crime.ds_info.json',
        'data_path': os.path.join(_RAW_DATA_DIR, 'crime.csv'),
        'task_type': 'regression',
        'n_classes': 1
    },
    'meps': {
        'ds_info_path': 'data/test_data/meps.ds_info.json',
        'data_path': os.path.join(_RAW_DATA_DIR, 'meps.csv'),
        'task_type': 'regression',
        'n_classes': 1
    }
}

def create_db_info_fz(dataset_name, config):
    """Create db_info_fz.json file for TabGNN."""
    print(f"\nProcessing dataset: {dataset_name}")

    with open(config['ds_info_path'], 'r', encoding='utf-8') as f:
        ds_info = json.load(f)

    df = pd.read_csv(config['data_path'], header=None)
    total = len(df)
    n_train = int(total * 0.8)
    n_test = total - n_train

    train_data = df.iloc[:n_train]
    if config['task_type'] == 'regression':
        train_class_counts = []
    else:
        target_col = len(ds_info['columns']) - 1
        train_targets = train_data.iloc[:, target_col]
        unique, counts = np.unique(train_targets, return_counts=True)
        train_class_counts = counts.tolist()

    db_info = {
        'task': {
            'type': config['task_type'],
            'n_classes': config['n_classes'],
            'n_train': n_train,
            'n_test': n_test,
            'train_class_counts': train_class_counts
        },
        'node_type_to_int': {
            'Main_table': 0
        },
        'edge_type_to_int': {
            'SELF': 0,
            'SIMILARITY_EDGE': 1
        },
        'node_types_and_features': {
            'Main_table': {}
        },
        'label_feature': 'Main_table.TARGET'
    }

    node_features = db_info['node_types_and_features']['Main_table']
    node_features['INDEX_ID'] = {'type': 'SCALAR'}

    for col_info in ds_info['columns']:
        if col_info['name'] == 'TARGET':
            if config['task_type'] == 'regression':
                node_features['TARGET'] = {'type': 'NUMERIC'}
            else:
                node_features['TARGET'] = {'type': 'CATEGORICAL', 'cardinality': col_info.get('cardinality', 2)}
        else:
            if col_info['type'] == 'NUMERIC':
                node_features[col_info['name']] = {'type': 'NUMERIC'}
            else:
                node_features[col_info['name']] = {'type': 'CATEGORICAL', 'cardinality': col_info['cardinality']}

    cat_features = [col['name'] for col in ds_info['columns'][:-1] if col['type'] == 'CATEGORICAL']
    edge_type_idx = 2
    for cat_feat in cat_features:
        db_info['edge_type_to_int'][cat_feat] = edge_type_idx
        edge_type_idx += 1

    db_info_path = f'data/{dataset_name}/{dataset_name}.db_info_fz.json'
    os.makedirs(os.path.dirname(db_info_path), exist_ok=True)
    with open(db_info_path, 'w', encoding='utf-8') as f:
        json.dump(db_info, f, indent=2, ensure_ascii=False)

    print(f"db_info_fz.json saved to: {db_info_path}")
    print(f"Train: {n_train} rows, Test: {n_test} rows")
    print(f"Features: {len(ds_info['columns']) - 1}")
    print(f"Connect keys: {len(cat_features)}")

    return db_info

if __name__ == '__main__':
    print("=" * 60)
    print("Creating db_info_fz.json for TabGNN")
    print("=" * 60)

    results = {}
    for dataset_name, config in DATASETS.items():
        if not os.path.exists(config['ds_info_path']):
            print(f"\nWarning: ds_info.json not found: {config['ds_info_path']}")
            continue
        if not os.path.exists(config['data_path']):
            print(f"\nWarning: data file not found: {config['data_path']}")
            continue
        try:
            db_info = create_db_info_fz(dataset_name, config)
            results[dataset_name] = 'success'
        except Exception as e:
            print(f"\nError processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("db_info_fz.json creation complete!")
    print("=" * 60)
