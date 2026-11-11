import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import your dataset loading functions
from datasets.adult_dataset import load_adult_reconstruct_dataset
from datasets.cardio_dataset import load_cardio_reconstruct_dataset
from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset
from datasets.diamonds_dataset import load_diamonds_reconstruct_dataset
from datasets.elevator_dataset import load_elevator_reconstruct_dataset
from datasets.housesale_dataset import load_housesale_reconstruct_dataset
from datasets.crime_dataset import load_crime_reconstruct_dataset
from datasets.meps_dataset import load_meps_reconstruct_dataset
from datasets.mslr_dataset import load_mslr_reconstruct_dataset_cls

def adapt_to_saint_format(dataset, v, num_classes_dict, target_col_idx=None, task='binary', test_size=0.2, val_size=0.15, random_state=42):
    """
    Adapt IndexedReconstructDataset to SAINT model format
    
    Args:
        dataset: IndexedReconstructDataset instance
        v: feature type vector, 0 for continuous features, 1 for categorical features
        num_classes_dict: dictionary of categorical feature classes
        target_col_idx: target column index, if None use last column
        task: task type ('binary', 'multiclass', 'regression')
        test_size: test set ratio
        val_size: validation set ratio
        random_state: random seed
    
    Returns:
        cat_dims: list of class numbers for each categorical feature
        cat_idxs: list of categorical feature column indices
        con_idxs: list of continuous feature column indices
        X_train, y_train, X_valid, y_valid, X_test, y_test: data dictionary
        train_mean, train_std: mean and std of continuous features
    """
    
    df = dataset.df.copy()
    
    if target_col_idx is None:
        target_col_idx = len(df.columns) - 1
    
    feature_cols = [i for i in range(len(df.columns)) if i != target_col_idx]
    X = df.iloc[:, feature_cols]
    y = df.iloc[:, target_col_idx]
    
    v_features = np.array([v[i] for i in feature_cols])
    
    cat_idxs = np.where(v_features == 1)[0].tolist()
    con_idxs = np.where(v_features == 0)[0].tolist()
    
    cat_dims = []
    for idx in cat_idxs:
        col_idx = feature_cols[idx]
        if col_idx in num_classes_dict:
            cat_dims.append(num_classes_dict[col_idx])
        else:
            unique_vals = X.iloc[:, idx].nunique()
            cat_dims.append(unique_vals)
    
    for idx in con_idxs:
        col = X.iloc[:, idx].copy()
        col.fillna(col.mean(), inplace=True)
        X.iloc[:, idx] = col
    
    for idx in cat_idxs:
        col = X.iloc[:, idx].copy()
        col = col.fillna("MissingValue")
        if col.dtype == 'object':
            le = LabelEncoder()
            col = le.fit_transform(col)
        X.iloc[:, idx] = col
    
    if task != 'regression':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    nan_mask = pd.DataFrame(1, index=X.index, columns=X.columns)
    
    total_samples = len(X)
    from utils.split_utils import split_indices
    train_indices, valid_indices, test_indices = split_indices(total_samples, seed=random_state)
    train_indices = train_indices.tolist()
    valid_indices = valid_indices.tolist()
    test_indices = test_indices.tolist()

    X_train = X.iloc[train_indices]
    X_valid = X.iloc[valid_indices]
    X_test = X.iloc[test_indices]
    
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_valid = y.iloc[valid_indices] if hasattr(y, 'iloc') else y[valid_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    nan_mask_train = nan_mask.iloc[train_indices]
    nan_mask_valid = nan_mask.iloc[valid_indices]
    nan_mask_test = nan_mask.iloc[test_indices]
    
    def to_saint_format(X, y, nan_mask):
        if hasattr(X, 'values'):
            X_data = X.values
        else:
            X_data = np.array(X)
        
        if hasattr(nan_mask, 'values'):
            nan_mask_data = nan_mask.values
        else:
            nan_mask_data = np.array(nan_mask)
        
        if hasattr(y, 'values'):
            y_data = y.values
        else:
            y_data = np.array(y)
        
        return {
            'data': X_data,
            'mask': nan_mask_data
        }, {
            'data': y_data.reshape(-1, 1)
        }
    
    X_train_dict, y_train_dict = to_saint_format(X_train, y_train, nan_mask_train)
    X_valid_dict, y_valid_dict = to_saint_format(X_valid, y_valid, nan_mask_valid)
    X_test_dict, y_test_dict = to_saint_format(X_test, y_test, nan_mask_test)
    
    if len(con_idxs) > 0:
        train_mean = np.array(X_train_dict['data'][:, con_idxs], dtype=np.float32).mean(0)
        train_std = np.array(X_train_dict['data'][:, con_idxs], dtype=np.float32).std(0)
        train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    else:
        train_mean = np.array([])
        train_std = np.array([])
    
    return cat_dims, cat_idxs, con_idxs, X_train_dict, y_train_dict, X_valid_dict, y_valid_dict, X_test_dict, y_test_dict, train_mean, train_std


class SaintDatasetAdapter(Dataset):
    """
    Adapt IndexedReconstructDataset to SAINT model format
    """
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):
        """
        Args:
            X: data dictionary {'data': array, 'mask': array}
            Y: target dictionary {'data': array}
            cat_cols: list of categorical feature indices
            task: task type ('clf' or 'reg')
            continuous_mean_std: mean and std of continuous features
        """
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        
        self.X1 = X[:, cat_cols].copy().astype(np.int64)
        self.X2 = X[:, con_cols].copy().astype(np.float32)
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)
        
        if task == 'clf':
            self.y = Y['data']
        else:
            self.y = Y['data'].astype(np.float32)
        
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        
        if continuous_mean_std is not None and len(continuous_mean_std[0]) > 0:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (np.concatenate((self.cls[idx], self.X1[idx])), 
                self.X2[idx], 
                self.y[idx], 
                np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), 
                self.X2_mask[idx])


# Example function to create SAINT datasets from IndexedReconstructDataset
def create_saint_datasets_from_reconstruct(dataset, v, num_classes_dict, target_col_idx=None, task='binary'):
    """
    Create SAINT model datasets from IndexedReconstructDataset
    
    Args:
        dataset: IndexedReconstructDataset instance
        v: feature type vector
        num_classes_dict: dictionary of categorical feature classes
        target_col_idx: target column index
        task: task type
    
    Returns:
        train_ds, valid_ds, test_ds: SAINT format datasets
        cat_dims, cat_idxs, con_idxs: feature information
        continuous_mean_std: continuous feature statistics
    """
    
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = adapt_to_saint_format(
        dataset, v, num_classes_dict, target_col_idx, task
    )
    
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
    
    train_ds = SaintDatasetAdapter(X_train, y_train, cat_idxs, 'clf' if task != 'regression' else 'reg', continuous_mean_std)
    valid_ds = SaintDatasetAdapter(X_valid, y_valid, cat_idxs, 'clf' if task != 'regression' else 'reg', continuous_mean_std)
    test_ds = SaintDatasetAdapter(X_test, y_test, cat_idxs, 'clf' if task != 'regression' else 'reg', continuous_mean_std)
    
    return train_ds, valid_ds, test_ds, cat_dims, cat_idxs, con_idxs, continuous_mean_std


# Dataset loading function mapping
DATASET_LOADERS = {
    'adult': load_adult_reconstruct_dataset,
    'cardio': load_cardio_reconstruct_dataset,
    'creditcard': load_creditcard_reconstruct_dataset,
    'diamonds': load_diamonds_reconstruct_dataset,
    'elevator': load_elevator_reconstruct_dataset,
    'housesale': load_housesale_reconstruct_dataset,
    'crime': load_crime_reconstruct_dataset,
    'meps': load_meps_reconstruct_dataset,
    'mslr': load_mslr_reconstruct_dataset_cls
}

def load_dataset_by_name(dataset_name, csv_path=None, max_samples=None):
    """
    Load dataset by name

    Args:
        dataset_name: dataset name ('adult', 'cardio', 'creditcard', 'diamonds', ...)
        csv_path: optional CSV file path, if not provided use default path
        max_samples: optional subsample size (only used for MSLR-WEB10K)

    Returns:
        dataset, v, num_classes_dict: dataset and related information
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_LOADERS.keys())}")

    loader_func = DATASET_LOADERS[dataset_name]

    if dataset_name == 'mslr':
        # 0 => full data (None), otherwise subsample
        ms = max_samples if max_samples else None
        dataset, v, num_classes_dict = loader_func(max_samples=ms)
    elif csv_path is None:
        dataset, v, num_classes_dict = loader_func()
    else:
        dataset, v, num_classes_dict = loader_func(csv_path)

    return dataset, v, num_classes_dict


def get_dataset_info(dataset_name):
    """
    Get basic dataset information
    
    Args:
        dataset_name: dataset name
    
    Returns:
        dict: dictionary containing dataset information
    """
    dataset_info = {
        'adult': {
            'description': 'Adult income dataset',
            'features': 14,
            'categorical': 0,
            'continuous': 14,
            'task': 'binary',
            'default_target': 14  # Last column is target
        },
        'cardio': {
            'description': 'Cardiovascular disease dataset',
            'features': 11,
            'categorical': 11,
            'continuous': 0,
            'task': 'binary',
            'default_target': 11  # Last column is target
        },
        'creditcard': {
            'description': 'Credit card fraud detection dataset',
            'features': 30,
            'categorical': 0,
            'continuous': 30,
            'task': 'binary',
            'default_target': 30  # Last column is target
        },
        'diamonds': {
            'description': 'Diamonds price prediction dataset',
            'features': 10,
            'categorical': 0,
            'continuous': 10,
            'task': 'regression',
            'default_target': 9  # Last column is target
        },
        'elevator': {
            'description': 'Elevator state regression (all continuous)',
            'features': 7,
            'categorical': 0,
            'continuous': 7,
            'task': 'regression',
            'default_target': 7  # Last column is target
        },
        'housesale': {
            'description': 'House sale price prediction (all categorical features)',
            'features': 39,
            'categorical': 39,
            'continuous': 0,
            'task': 'regression',
            'default_target': 39  # Last column is target
        },
        'crime': {
            'description': 'Crime and communities (regression)',
            'features': 122,
            'categorical': 3,
            'continuous': 119,
            'task': 'regression',
            'default_target': None  # last column
        },
        'meps': {
            'description': 'Medical Expenditure Panel Survey (regression)',
            'features': 138,
            'categorical': 134,
            'continuous': 4,
            'task': 'regression',
            'default_target': None  # last column
        },
        'mslr': {
            'description': 'MSLR-WEB10K (binary classification)',
            'features': 136,
            'categorical': 0,
            'continuous': 136,
            'task': 'binary',
            'default_target': None  # last column
        }
    }
    
    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_info[dataset_name] 