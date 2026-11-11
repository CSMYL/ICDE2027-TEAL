import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA = os.path.join(PROJECT_ROOT, 'raw_data')

CLASSIFICATION = {'creditcard', 'cardio', 'adult'}
# Regression datasets whose target is standardized by the CAIM loaders
STANDARDIZE_TARGET = {'diamonds', 'elevator', 'housesale'}


def load_data(dataset):
    """Read the raw CSV directly and return (X_features_df, y, is_classification).

    Target handling matches the CAIM loaders:
    - classification: integer 0/1 labels
    - regression (diamonds/elevator/housesale): standardized target
    - regression (crime/meps): raw target (already standardized in the CSV)
    """
    p = os.path.join(RAW_DATA, f'{dataset}.csv')

    if dataset in ('creditcard', 'cardio', 'adult', 'diamonds'):
        df = pd.read_csv(p, header=None)
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
    elif dataset in ('elevator', 'housesale'):
        df = pd.read_csv(p, header=0)
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
    elif dataset == 'crime':
        df = pd.read_csv(p, header=0)
        y = df['ViolentCrimesPerPop'].values.astype(float)
        X = df.drop(columns=['ViolentCrimesPerPop'])
    elif dataset == 'meps':
        df = pd.read_csv(p, header=0)
        drop_cols = [c for c in df.columns if 'Unnamed' in c or 'PERWT' in c]
        df = df.drop(columns=drop_cols)
        y = df['UTILIZATION_reg'].values.astype(float)
        X = df.drop(columns=['UTILIZATION_reg'])
    else:
        raise ValueError(dataset)

    X = X.astype(float)
    if dataset in CLASSIFICATION:
        y = y.astype(np.int64)
        return X, y, True
    else:
        y = y.astype(float)
        if dataset in STANDARDIZE_TARGET:
            y = (y - y.mean()) / y.std()
        return X, y, False


def main():
    parser = argparse.ArgumentParser(description='XTab baseline: FT-Transformer fine-tuned on cross-table pretrained weights')
    parser.add_argument('--dataset', required=True, choices=['creditcard', 'cardio', 'adult', 'diamonds', 'elevator', 'housesale', 'crime', 'meps'])
    parser.add_argument('--pretrained_ckpt', type=str, default=os.path.join(PROJECT_ROOT, 'baseline', 'xtab', 'checkpoints', 'iter_2k.ckpt'))
    parser.add_argument('--max_samples', type=int, default=0, help='subsample size (0 = full data)')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--time_limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    np.random.seed(args.seed)

    X, y, is_cls = load_data(args.dataset)
    logger.info(f'Dataset: {args.dataset}, X: {X.shape}')

    # Subsample (transformer is slow on large data)
    n = len(X)
    if args.max_samples > 0 and n > args.max_samples:
        idx = np.random.choice(n, args.max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        logger.info(f'Subsampled to {len(X)} samples')

    # Split train/test 8:2
    from utils.split_utils import split_indices
    train_idx, _, test_idx = split_indices(len(X), seed=args.seed)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Build AutoGluon DataFrames with string column names
    train_df = X_train.copy()
    train_df.columns = [str(c) for c in train_df.columns]
    train_df['label'] = y_train
    test_df = X_test.copy()
    test_df.columns = [str(c) for c in test_df.columns]
    test_df['label'] = y_test
    logger.info(f'Train/Test: {len(train_df)}/{len(test_df)}')

    hyperparameters = {
        'FT_TRANSFORMER': {
            'finetune_on': args.pretrained_ckpt,
            'optimization.max_epochs': args.max_epochs,
            'env.per_gpu_batch_size': args.batch_size,
            'env.num_workers': 0,
            'env.num_workers_evaluation': 0,
        }
    }
    eval_metric = 'roc_auc' if is_cls else 'root_mean_squared_error'

    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(label='label', eval_metric=eval_metric)
    logger.info(f'Training XTab (ckpt={args.pretrained_ckpt}, metric={eval_metric}, epochs={args.max_epochs})...')

    fit_kwargs = dict(
        train_data=train_df,
        hyperparameters=hyperparameters,
        keep_only_best=True,
        fit_weighted_ensemble=False,
    )
    if args.time_limit is not None:
        fit_kwargs['time_limit'] = args.time_limit

    predictor.fit(**fit_kwargs)

    results = predictor.evaluate(test_df, silent=True)
    logger.info(f'{args.dataset} (XTab): {results}')


if __name__ == '__main__':
    main()
