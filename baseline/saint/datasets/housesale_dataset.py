import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np
import os

def load_housesale_reconstruct_dataset(csv_path=None):
    """
    Load housesale dataset. All columns except the last are categorical features, the last column is a regression target.
    Returns: dataset, v, num_classes_dict
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        csv_path = os.path.join(project_root, 'raw_data', 'housesale.csv')
    df = pd.read_csv(csv_path, header=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_features = len(df.columns)
    cat_indices = list(range(num_features - 1))
    cont_indices = [num_features - 1]
    label_encoders = {}
    for col_idx in cat_indices:
        le = LabelEncoder()
        df.iloc[:, col_idx] = le.fit_transform(df.iloc[:, col_idx])
        label_encoders[col_idx] = le
    v = np.ones(num_features, dtype=np.int64)
    v[cont_indices] = 0
    num_classes_dict = {col_idx: len(label_encoders[col_idx].classes_) for col_idx in cat_indices}
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict 