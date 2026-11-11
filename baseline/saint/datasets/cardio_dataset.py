import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_cardio_reconstruct_dataset(csv_path=None):
    """
    Load cardio dataset and construct IndexedReconstructDataset and v vector.
    Cardio dataset has 12 features:
    - All 12 features are categorical features

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray, vector of length F, 0 for continuous features, 1 for categorical features
        num_classes_dict: dict, keys are categorical feature indices, values are number of classes
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        csv_path = os.path.join(project_root, 'raw_data', 'cardio.csv')
    df = pd.read_csv(csv_path, header=None)
    
    num_features = len(df.columns)
    cont_indices = []
    cat_indices = list(range(num_features - 1))

    label_encoders = {}
    for col_idx in cat_indices:
        label_encoder = LabelEncoder()
        df.iloc[:, col_idx] = label_encoder.fit_transform(df.iloc[:, col_idx])
        label_encoders[col_idx] = label_encoder

    v = np.ones(num_features, dtype=np.int64)
    v[-1] = 0

    num_classes_dict = {}
    for col_idx in cat_indices:
        num_classes_dict[col_idx] = len(label_encoders[col_idx].classes_)

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 