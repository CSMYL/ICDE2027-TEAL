import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
import logging
import time
import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.baseline_datasets import (
    load_creditcard_baseline_dataset,
    load_adult_baseline_dataset,
    load_cardio_baseline_dataset,
    load_diamonds_baseline_dataset,
    load_elevator_baseline_dataset,
    load_housesale_baseline_dataset,
    load_crime_baseline_dataset,
    load_meps_baseline_dataset,
)

DATASET_LOADERS = {
    'creditcard': load_creditcard_baseline_dataset,
    'adult': load_adult_baseline_dataset,
    'cardio': load_cardio_baseline_dataset,
    'cardiovascular': load_cardio_baseline_dataset,
    'diamonds': load_diamonds_baseline_dataset,
    'elevator': load_elevator_baseline_dataset,
    'housesale': load_housesale_baseline_dataset,
    'crime': load_crime_baseline_dataset,
    'meps': load_meps_baseline_dataset,
}


def convert_dataset_to_numpy(dataset, v, target_idx):
    all_data, all_targets = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_data.append(x.numpy())
        all_targets.append(y.item())
    return np.array(all_data), np.array(all_targets)


def test_orion_bix_baseline(
    dataset='creditcard',
    n_estimators=32,
    batch_size=8,
    n_bins=10,          # number of bins for regression discretization
    gpu_id=-1,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')
    logger.info(f'Parameters: dataset={dataset}, n_estimators={n_estimators}, n_bins={n_bins}')

    key = dataset.lower()
    if key not in DATASET_LOADERS:
        raise ValueError(
            f'Unsupported dataset: {dataset}, choices: {sorted(set(DATASET_LOADERS))}'
        )

    dataset_obj, v, num_classes_dict, target_idx = DATASET_LOADERS[key]()
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)

    is_regression = v[target_idx].item() == 0
    task_type = 'regression' if is_regression else 'classification'
    logger.info(f'Dataset: {key}, task: {task_type}, size: {len(dataset_obj)}, target idx: {target_idx}')

    # Unified split: deterministic random shuffle -> 80/20 train-val/test -> 80/20 train/val
    from utils.split_utils import split_indices
    total_size = len(dataset_obj)
    train_idx, val_idx, test_idx = split_indices(total_size, seed=seed)
    train_dataset = torch.utils.data.Subset(dataset_obj, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(dataset_obj, val_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset_obj, test_idx.tolist())
    logger.info(f'Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}')

    # Convert to numpy
    train_data, train_targets = convert_dataset_to_numpy(train_dataset, v, target_idx)
    val_data, val_targets = convert_dataset_to_numpy(val_dataset, v, target_idx)
    test_data, test_targets = convert_dataset_to_numpy(test_dataset, v, target_idx)

    # Combine train+val for Orion-BiX fit (ICL doesn't use val set for early stopping)
    X_train = np.concatenate([train_data, val_data], axis=0)
    y_train = np.concatenate([train_targets, val_targets], axis=0)
    X_test = test_data
    y_test = test_targets

    logger.info(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X_train.shape[1]}')

    # Import Orion-BiX
    try:
        from orion_bix.sklearn.classifier import OrionBixClassifier
    except ImportError:
        logger.error("Orion-BiX not installed. Run: pip install orion-bix")
        return

    # ---------- Regression: target discretization ----------
    binner = None
    bin_centers = None
    y_train_cls = y_train
    y_test_cls = y_test

    if is_regression:
        logger.info(f'Regression detected — discretizing target into {n_bins} quantile bins...')

        # Fit binner on train targets only
        y_train_2d = y_train.reshape(-1, 1)
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile',
                                  random_state=seed, subsample=200000)
        y_train_cls = binner.fit_transform(y_train_2d).astype(np.int64).ravel()
        y_test_cls = binner.transform(y_test.reshape(-1, 1)).astype(np.int64).ravel()

        # Compute bin centers for reconstruction
        bin_edges = binner.bin_edges_[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        n_unique = len(np.unique(y_train_cls))
        logger.info(f'  Bins: {n_unique} unique classes, centers: '
                    f'{[f"{c:.3f}" for c in bin_centers[:5]]}...')

    # ---------- Fit ----------
    logger.info('Fitting Orion-BiX (first run downloads ~800MB checkpoint from HuggingFace)...')
    clf = OrionBixClassifier(
        n_estimators=n_estimators,
        batch_size=batch_size,
        device=device,
        random_state=seed,
        verbose=False,
    )

    t0 = time.time()
    clf.fit(X_train, y_train_cls)
    fit_time = time.time() - t0
    logger.info(f'Fit time: {fit_time:.1f}s')

    # ---------- Predict ----------
    t0 = time.time()
    y_proba = clf.predict_proba(X_test)
    infer_time = time.time() - t0
    logger.info(f'Inference time: {infer_time:.2f}s ({len(X_test)} samples)')

    # ---------- Metrics ----------
    if is_regression:
        # Reconstruct continuous predictions from class probabilities
        # y_proba shape: (n_samples, n_bins) — weighted average of bin centers
        n_classes_actual = y_proba.shape[1]
        if n_classes_actual < len(bin_centers):
            # Some bins may be empty in training data → pad/align
            bin_centers = bin_centers[:n_classes_actual]
        y_pred_cont = y_proba @ bin_centers[:n_classes_actual]

        mse = mean_squared_error(y_test, y_pred_cont)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_cont)
        logger.info(f'Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}, Test R²: {r2:.4f}')
    else:
        y_pred = np.argmax(y_proba, axis=1)
        acc = accuracy_score(y_test_cls, y_pred)
        try:
            auc = roc_auc_score(y_test_cls, y_proba[:, 1])
        except ValueError:
            auc = float('nan')
        logger.info(f'Test ACC: {acc:.4f}, Test AUC: {auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orion-BiX Baseline (Classification + Regression via Binning)')
    parser.add_argument('--dataset', type=str, default='creditcard',
                        choices=sorted(set(DATASET_LOADERS.keys())),
                        help='Dataset name')
    parser.add_argument('--n_estimators', type=int, default=32,
                        help='Number of ensemble members')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for ICL inference')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of quantile bins for regression discretization')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_orion_bix_baseline(
        dataset=args.dataset,
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )
