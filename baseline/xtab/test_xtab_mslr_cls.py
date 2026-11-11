import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.mslr_loader import load_mslr_cls

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CKPT = os.path.join(PROJECT_ROOT, 'baseline', 'xtab', 'checkpoints', 'iter_2k.ckpt')


def main():
    parser = argparse.ArgumentParser(description='XTab on MSLR-WEB10K (binary classification)')
    parser.add_argument('--pretrained_ckpt', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--max_samples', type=int, default=100000, help='0 = full data')
    parser.add_argument('--threshold', type=int, default=0,
                        help='relevance > threshold => positive (default 0)')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    np.random.seed(args.seed)

    X, y = load_mslr_cls(
        max_samples=args.max_samples if args.max_samples > 0 else None,
        binary=True, threshold=args.threshold,
    )
    logger.info(f'MSLR XTab-cls: X {X.shape}, positive rate {y.mean():.3f}')

    from utils.split_utils import split_indices
    train_idx, _, test_idx = split_indices(len(X), seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cols = [f'f{i}' for i in range(X.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test, columns=cols)
    test_df['label'] = y_test

    hyperparameters = {
        'FT_TRANSFORMER': {
            'finetune_on': args.pretrained_ckpt,
            'optimization.max_epochs': args.max_epochs,
            'env.per_gpu_batch_size': args.batch_size,
            'env.num_workers': 0,
            'env.num_workers_evaluation': 0,
        }
    }

    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(label='label', eval_metric='roc_auc')
    predictor.fit(
        train_data=train_df, hyperparameters=hyperparameters,
        keep_only_best=True, fit_weighted_ensemble=False,
    )
    results = predictor.evaluate(test_df, silent=True)
    logger.info(f'XTab MSLR-cls: {results}')


if __name__ == '__main__':
    main()
