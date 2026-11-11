"""Unified dataset split used by CAIM and all baselines.

The scheme is: deterministic random shuffle -> 80/20 train-val/test -> then 20% of
train-val as validation. This gives train 64% / val 16% / test 20%.

Random shuffling (instead of a sequential first-80%/last-20% cut) removes any
ordering bias, which matters for query-grouped datasets such as MSLR-WEB10K.
"""

import numpy as np


def split_indices(n, seed=42, test_ratio=0.2, val_ratio=0.2):
    """Deterministically split n sample indices into (train, val, test).

    Args:
        n: total number of samples.
        seed: random seed (fixed => reproducible split).
        test_ratio: fraction held out for test (default 0.2).
        val_ratio: fraction of the train+val block used for validation (default 0.2).

    Returns:
        (train_idx, val_idx, test_idx): numpy int64 arrays of positions into the
        original (unshuffled) data.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(test_ratio * n)
    train_val = perm[: n - n_test]
    test = perm[n - n_test:]

    n_val = int(val_ratio * len(train_val))
    val = train_val[:n_val]
    train = train_val[n_val:]

    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)
