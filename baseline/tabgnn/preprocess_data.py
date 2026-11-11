import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Dataset configs matching the CAIM paper datasets section
DATASET_CONFIGS = {
    'adult': {
        'task': 'binary classification',
        'continuous_cols': ['age', 'fnlwgt', 'education-num', 'capital-gain',
                            'capital-loss', 'hours-per-week'],
        'categorical_cols': ['workclass', 'education', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'native-country'],
    },
    'cardio': {
        'task': 'binary classification',
        'continuous_cols': ['age', 'height', 'weight', 'ap_hi', 'ap_lo'],
        'categorical_cols': ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
    },
    'creditcard': {
        'task': 'binary classification',
        'continuous_cols': [f'V{i}' for i in range(1, 29)] + ['Amount'],
        'categorical_cols': [],
    },
    'diamonds': {
        'task': 'regression',
        'continuous_cols': ['carat', 'depth', 'table', 'x', 'y', 'z'],
        'categorical_cols': ['cut', 'color', 'clarity'],
    },
    'elevator': {
        'task': 'regression',
        'continuous_cols': [f'feature{i}' for i in range(7)],
        'categorical_cols': [],
    },
    'housesale': {
        'task': 'regression',
        'continuous_cols': [],
        'categorical_cols': [f'feature{i}' for i in range(40)],
    },
    'crime': {
        'task': 'regression',
        'continuous_cols': [f'feature{i}' for i in range(119)],
        'categorical_cols': [f'cat_feature{i}' for i in range(3)],
    },
    'meps': {
        'task': 'regression',
        'continuous_cols': ['AGE', 'PCS42', 'MCS42', 'K6SUM42'],
        'categorical_cols': [f'cat_feature{i}' for i in range(134)],
    },
}


def generate_ds_info(csv_path, dataset_name):
    """Generate ds_info.json for a dataset by scanning the CSV."""
    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    columns = []
    for col in feature_cols:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            cardinality = int(df[col].nunique())
            columns.append({
                'name': str(col),
                'type': 'CATEGORICAL',
                'cardinality': cardinality,
            })
        else:
            columns.append({
                'name': str(col),
                'type': 'NUMERIC',
            })

    ds_info = {
        'dataset_name': dataset_name,
        'columns': columns,
        'label_column': {
            'name': str(target_col),
            'type': 'NUMERIC' if DATASET_CONFIGS.get(dataset_name, {}).get('task') == 'regression' else 'CATEGORICAL',
        },
    }
    return ds_info


def main():
    raw_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'raw_data'
    )
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_data')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Raw data dir: {raw_data_dir}')
    print(f'Output dir:   {output_dir}')
    print()

    for name, config in DATASET_CONFIGS.items():
        csv_path = os.path.join(raw_data_dir, f'{name}.csv')
        if not os.path.exists(csv_path):
            print(f'SKIP {name}: {csv_path} not found')
            continue

        print(f'Generating ds_info for {name} ({config["task"]})...')
        ds_info = generate_ds_info(csv_path, name)

        output_path = os.path.join(output_dir, f'{name}.ds_info.json')
        with open(output_path, 'w') as f:
            json.dump(ds_info, f, indent=2)

        n_cat = sum(1 for c in ds_info['columns'] if c['type'] == 'CATEGORICAL')
        n_num = sum(1 for c in ds_info['columns'] if c['type'] == 'NUMERIC')
        print(f'  {n_num} numeric + {n_cat} categorical -> {output_path}')

    print('\nDone. Config files written to baseline/tabgnn/data/test_data/')
    print('To run TabGNN: cd baseline/tabgnn && python train.py --dataset adult --epochs 10')


if __name__ == '__main__':
    main()
