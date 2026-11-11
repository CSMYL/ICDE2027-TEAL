import os
import sys
import argparse
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset
from datasets.cardio_dataset import load_cardio_reconstruct_dataset
from datasets.adult_dataset import load_adult_reconstruct_dataset
from datasets.diamonds_dataset import load_diamonds_reconstruct_dataset
from datasets.elevator_dataset import load_elevator_reconstruct_dataset
from datasets.housesale_dataset import load_housesale_reconstruct_dataset
from datasets.crime_dataset import load_crime_reconstruct_dataset
from datasets.meps_dataset import load_meps_reconstruct_dataset


DATASET_LOADERS = {
    'creditcard': load_creditcard_reconstruct_dataset,
    'cardio': load_cardio_reconstruct_dataset,
    'adult': load_adult_reconstruct_dataset,
    'diamonds': load_diamonds_reconstruct_dataset,
    'elevator': load_elevator_reconstruct_dataset,
    'housesale': load_housesale_reconstruct_dataset,
    'crime': load_crime_reconstruct_dataset,
    'meps': load_meps_reconstruct_dataset,
}

CLASSIFICATION = {'creditcard', 'cardio', 'adult'}


def main():
    parser = argparse.ArgumentParser(description='TabPFN baseline')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='subsample size (TabPFN pretraining limit ~10k; 0 = full data)')
    parser.add_argument('--gpu_id', type=int, default=-1, help='-1 = auto device')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    np.random.seed(args.seed)

    # Load data
    dataset_obj, v, num_classes_dict = DATASET_LOADERS[args.dataset]()
    target_idx = len(v) - 1

    # Convert to X (features, drop target) and y (target)
    X, y = [], []
    for i in range(len(dataset_obj)):
        x, _ = dataset_obj[i]
        xi = x.numpy()
        X.append(xi[:-1])
        y.append(xi[-1])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    logger.info(f'Dataset: {args.dataset}, X: {X.shape}, y: {y.shape}')

    # Subsample (TabPFN pretraining limits)
    n = len(X)
    if args.max_samples > 0 and n > args.max_samples:
        idx = np.random.choice(n, args.max_samples, replace=False)
        X, y = X[idx], y[idx]
        logger.info(f'Subsampled to {len(X)} samples')

    # Split train/test 8:2
    from utils.split_utils import split_indices
    train_idx, _, test_idx = split_indices(len(X), seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f'Train/Test: {len(X_train)}/{len(X_test)}')

    device = f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'auto'

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    if args.dataset in CLASSIFICATION:
        y_train = y_train.astype(np.int64)
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device=device, random_state=args.seed)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, (y_proba > 0.5).astype(int))
        logger.info(f'{args.dataset} (TabPFN): AUC = {auc:.4f}, Acc = {acc:.4f}')
    else:
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device=device, random_state=args.seed)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        logger.info(f'{args.dataset} (TabPFN): RMSE = {rmse:.4f}, R2 = {r2:.4f}')


if __name__ == '__main__':
    main()
