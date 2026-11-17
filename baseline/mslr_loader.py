import os
import numpy as np
import pyarrow.parquet as pq

_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'raw_data', 'mslr')


def load_mslr_xy(max_samples=None, standardize=True):
    """Flatten the grouped MSLR parquet into (X, y).

    X: [n, 136] continuous features
    y: [n] relevance label (0-4), standardized by default so regression RMSE is comparable
    """
    parquet_path = os.path.join(_RAW, 'train.parquet')
    X_rows, y_rows = [], []
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
        X, y = X[:max_samples], y[:max_samples]

    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

    return X, y


def load_mslr_cls(max_samples=None, binary=True, threshold=0):
    """Load MSLR-WEB10K for classification.

    X is standardized; y is binary (relevance > threshold) by default, or the
    5-class relevance 0..4 if binary=False.
    """
    X, y = load_mslr_xy(max_samples=max_samples, standardize=False)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # standardize features only
    y = y.astype(np.int64)
    if binary:
        y = (y > threshold).astype(np.int64)  # 0 = irrelevant, 1 = relevant
    return X, y
