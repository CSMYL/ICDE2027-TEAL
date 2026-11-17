import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_adult_reconstruct_dataset(csv_path=None):
    """
    Load adult dataset and construct IndexedReconstructDataset and v vector.
    Adult dataset has 14 features:
    - First 13 features are continuous features (already standardized)
    - Last feature (income) is categorical feature

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray, vector of length F, 0 for continuous features, 1 for categorical features
        num_classes_dict: dict, keys are categorical feature indices, values are number of classes
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        csv_path = os.path.join(project_root, 'raw_data', 'adult.csv')
    df = pd.read_csv(csv_path, header=None)
    
    num_features = len(df.columns)
    cont_indices = list(range(num_features - 1))
    cat_indices = []

    v = np.zeros(num_features, dtype=np.int64)

    num_classes_dict = {}

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 