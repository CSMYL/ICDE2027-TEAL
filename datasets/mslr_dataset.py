import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from datasets import IndexedReconstructDataset

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
_RAW_DATA = os.path.join(_PROJECT_ROOT, 'raw_data', 'mslr')


def _flatten_parquet(parquet_path, max_samples=None):
    """Flatten the grouped (query -> documents) MSLR parquet into (X, y) numpy arrays.

    Each query row has `labels` (relevance per document) and `features` (136-dim feature
    vector per document). We expand each query into one row per document.
    """
    X_rows = []
    y_rows = []
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=200):
        df = batch.to_pandas()
        for labels, features in zip(df['labels'], df['features']):
            for lbl, feat in zip(labels, features):
                y_rows.append(lbl)
                X_rows.append(feat)
                if max_samples is not None and len(y_rows) >= max_samples:
                    break
            if max_samples is not None and len(y_rows) >= max_samples:
                break
        if max_samples is not None and len(y_rows) >= max_samples:
            break

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]
    return X, y


def load_mslr_reconstruct_dataset(parquet_path=None, max_samples=None, standardize=True):
    """Load MSLR-WEB10K as a reconstruction dataset.

    Returns (dataset, v, num_classes_dict) where:
      - dataset: IndexedReconstructDataset (X = 136 features + relevance target last)
      - v: all 0 (all features continuous; relevance treated as a regression target)
      - num_classes_dict: {} (no categorical features)

    max_samples: subsample the number of documents (None = use all ~723k train docs).
    """
    if parquet_path is None:
        parquet_path = os.path.join(_RAW_DATA, 'train.parquet')

    X, y = _flatten_parquet(parquet_path, max_samples=max_samples)

    if standardize:
        # Standardize features (column-wise) and the relevance target
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

    F = X.shape[1]
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(F)])
    df['target'] = y

    num_features = F + 1
    v = np.zeros(num_features, dtype=np.int64)  # all continuous (relevance = regression)
    num_classes_dict = {}

    dataset = IndexedReconstructDataset(df, cont_indices=list(range(num_features)), cat_indices=[])

    return dataset, v, num_classes_dict


def load_mslr_reconstruct_dataset_cls(parquet_path=None, max_samples=None, binary=True, threshold=0):
    """Load MSLR-WEB10K as a classification task.

    binary=True (default): map relevance 0..4 to binary labels
        y = (relevance > threshold), so 0 = irrelevant, 1 = relevant. num_classes = 2.
    binary=False: keep the 5-class relevance 0..4. num_classes = 5.

    Returns (dataset, v, num_classes_dict) where:
      - dataset: IndexedReconstructDataset (136 continuous features + integer class target last)
      - v: 0 for the 136 continuous features, 1 for the categorical target
      - num_classes_dict: {target_idx: 2} (binary) or {target_idx: 5} (5-class)
    """
    if parquet_path is None:
        parquet_path = os.path.join(_RAW_DATA, 'train.parquet')

    X, y = _flatten_parquet(parquet_path, max_samples=max_samples)

    # Standardize features only; keep relevance as integer class labels
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = y.astype(np.int64)

    if binary:
        y = (y > threshold).astype(np.int64)  # 0 = irrelevant, 1 = relevant
        num_classes = 2
    else:
        num_classes = 5  # relevance 0..4 (fixed, regardless of subsample)

    F = X.shape[1]
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(F)])
    df['target'] = y

    target_idx = F
    num_features = F + 1
    v = np.zeros(num_features, dtype=np.int64)
    v[target_idx] = 1  # categorical target

    num_classes_dict = {target_idx: num_classes}

    dataset = IndexedReconstructDataset(
        df, cont_indices=list(range(F)), cat_indices=[target_idx]
    )

    return dataset, v, num_classes_dict


if __name__ == '__main__':
    import sys
    ds, v, nd = load_mslr_reconstruct_dataset(max_samples=10000)
    x, y = ds[0]
    print('v shape:', v.shape)
    print('num_classes_dict:', nd)
    print('sample x shape:', x.shape, 'dtype:', x.dtype)
    print('total samples:', len(ds))
