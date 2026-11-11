import pandas as pd
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_crime_reconstruct_dataset(csv_path=None):
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'crime.csv')

    df = pd.read_csv(csv_path, header=0)

    target_col = 'ViolentCrimesPerPop'
    if df.columns[-1] != target_col:
        cols = [col for col in df.columns if col != target_col] + [target_col]
        df = df[cols]

    num_features = len(df.columns)

    # Per crime_info.json: first 119 features are continuous, next 3 are categorical,
    # last is the regression target (continuous). Total: 119 + 3 + 1 = 122 cols.
    cont_indices = []
    cat_indices = []

    for col_idx in range(num_features):
        col_name = df.columns[col_idx]
        if col_name == target_col:
            cont_indices.append(col_idx)
        elif col_idx >= num_features - 4 and col_idx < num_features - 1:
            cat_indices.append(col_idx)
        else:
            cont_indices.append(col_idx)

    v = np.zeros(num_features, dtype=np.int64)
    for cat_idx in cat_indices:
        v[cat_idx] = 1

    num_classes_dict = {}
    for col_idx in cat_indices:
        unique_vals = df.iloc[:, col_idx].nunique()
        num_classes_dict[col_idx] = unique_vals

    for col_idx in cat_indices:
        col_data = df.iloc[:, col_idx]
        if not pd.api.types.is_integer_dtype(col_data):
            df.iloc[:, col_idx] = col_data.round().astype(int)

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict
