import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_elevator_reconstruct_dataset(csv_path=None):
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'elevator.csv')
    # Read data (first row is header)
    df = pd.read_csv(csv_path, header=0)

    # Regression dataset: all features continuous, last column is the continuous target
    num_features = len(df.columns)
    cont_indices = list(range(num_features))  # All columns continuous (last = regression target)
    cat_indices = []  # No categorical features

    # Cast to float then standardize all features (including target)
    df = df.astype(float)
    scaler = StandardScaler()
    df.iloc[:, cont_indices] = scaler.fit_transform(df.iloc[:, cont_indices])

    # v vector: 0 = continuous, 1 = categorical
    v = np.zeros(num_features, dtype=np.int64)

    # No categorical features
    num_classes_dict = {}

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict
