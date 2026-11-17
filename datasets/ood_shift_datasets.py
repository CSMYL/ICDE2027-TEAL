import pandas as pd
import numpy as np
import os

from datasets import IndexedReconstructDataset


def _load_shift_reconstruct_dataset(csv_path: str, shuffle: bool = True):
    """
    Generic loader for diamonds OOD shift datasets (shift1/2/3).

    Assumes columns: carat, cut, color, clarity, depth, table, x, y, z, price
    Where:
        - Continuous: carat, depth, table, x, y, z, price (already standardized during generation)
        - Categorical: cut, color, clarity (integer-encoded per diamonds_mapping.csv)

    shuffle: if True, shuffle rows (for random splits); if False, preserve CSV row order
             (for OOD: first N_train rows = train, last N_test rows = test).

    Returns:
        dataset: IndexedReconstructDataset
        v: (F,) 0/1 vector, 1 for categorical features
        num_classes_dict: {cat_idx: num_classes}
    """
    df = pd.read_csv(csv_path, header=0)
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    num_features = len(df.columns)
    cat_indices = [1, 2, 3]
    cont_indices = [i for i in range(num_features) if i not in cat_indices]

    v = np.zeros(num_features, dtype=np.int64)
    v[cat_indices] = 1

    num_classes_dict = {
        idx: int(df.iloc[:, idx].nunique())
        for idx in cat_indices
    }

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict


def load_shift1_reconstruct_dataset(csv_path: str = None):
    """Load shift1 dataset (Cut Shift OOD experiment)."""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'diamonds_ood_experiments', 'shift1.csv')
    return _load_shift_reconstruct_dataset(csv_path)


def load_shift2_reconstruct_dataset(csv_path: str = None):
    """Load shift2 dataset (Color Shift OOD experiment)."""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'diamonds_ood_experiments', 'shift2.csv')
    return _load_shift_reconstruct_dataset(csv_path)


def load_shift3_reconstruct_dataset(csv_path: str = None):
    """Load shift3 dataset (Simpson's Paradox OOD experiment)."""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'diamonds_ood_experiments', 'shift3.csv')
    return _load_shift_reconstruct_dataset(csv_path)
