import pandas as pd
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_meps_reconstruct_dataset(csv_path=None):
    """
    Load meps dataset and construct IndexedReconstructDataset and v vector.
    MEPS has 138 features:
    - 4 continuous features (already standardized): AGE, PCS42, MCS42, K6SUM42
    - 134 categorical features (one-hot encoded)
    - Last column is regression target: UTILIZATION_reg
    - Note: data may contain Unnamed: 0 and PERWT15F columns that need to be dropped.

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray of length F, 0 for continuous, 1 for categorical
        num_classes_dict: dict mapping categorical feature indices to their cardinality
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'meps.csv')

    df = pd.read_csv(csv_path, header=0)

    columns_to_drop = []
    for col in df.columns:
        if 'Unnamed' in col or 'PERWT' in col:
            columns_to_drop.append(col)

    df = df.drop(columns=columns_to_drop)

    target_col = 'UTILIZATION_reg'
    if df.columns[-1] != target_col:
        cols = [col for col in df.columns if col != target_col] + [target_col]
        df = df[cols]

    num_features = len(df.columns)

    continuous_feature_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', target_col]
    cont_indices = []
    cat_indices = []

    for col_idx, col_name in enumerate(df.columns):
        if col_name in continuous_feature_names:
            cont_indices.append(col_idx)
        else:
            cat_indices.append(col_idx)

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
