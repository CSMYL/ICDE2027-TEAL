import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_cardio_reconstruct_dataset(csv_path=None):
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'cardio.csv')
    # Read data (no column names)
    df = pd.read_csv(csv_path, header=None)
    
    # Define continuous and categorical features by position
    num_features = len(df.columns)
    cont_indices = list(range(num_features - 1))  # All except last column are continuous features
    cat_indices = [num_features - 1]  # Last column is categorical feature

    # Standardize continuous features
    scaler = StandardScaler()
    df.iloc[:, cont_indices] = scaler.fit_transform(df.iloc[:, cont_indices])
    
    # Tokenize categorical features
    label_encoder = LabelEncoder()
    df.iloc[:, cat_indices[0]] = label_encoder.fit_transform(df.iloc[:, cat_indices[0]])

    # Construct v vector: 0 indicates continuous feature, 1 indicates categorical feature
    v = np.zeros(num_features, dtype=np.int64)
    v[cat_indices] = 1

    # Construct num_classes_dict
    num_classes_dict = {
        cat_indices[0]: len(label_encoder.classes_)
    }

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 