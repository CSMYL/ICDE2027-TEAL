import pandas as pd
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_diamonds_reconstruct_dataset(csv_path=None):
    """
    Load diamonds dataset and construct IndexedReconstructDataset and v vector.
    Diamonds dataset has 10 features:
    - All 10 features are continuous features (already standardized)

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray, vector of length F, 0 for continuous features, 1 for categorical features
        num_classes_dict: dict, keys are categorical feature indices, values are number of classes
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        csv_path = os.path.join(project_root, 'raw_data', 'diamonds.csv')
    df = pd.read_csv(csv_path, header=None)
    
    num_features = len(df.columns)
    cont_indices = list(range(num_features - 1))
    cat_indices = []

    v = np.zeros(num_features, dtype=np.int64)

    num_classes_dict = {}

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 