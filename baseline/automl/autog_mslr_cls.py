import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.mslr_loader import load_mslr_cls


def main():
    parser = argparse.ArgumentParser(description='AutoGluon on MSLR-WEB10K (binary classification)')
    parser.add_argument('--max_samples', type=int, default=100000, help='0 = full ~723k docs')
    parser.add_argument('--threshold', type=int, default=0,
                        help='relevance > threshold => positive (default 0)')
    parser.add_argument('--time_limit', type=int, default=600, help='AutoGluon fit time limit (seconds)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X, y = load_mslr_cls(
        max_samples=args.max_samples if args.max_samples > 0 else None,
        binary=True, threshold=args.threshold,
    )
    print(f'MSLR AutoGluon-cls: X {X.shape}, positive rate {y.mean():.3f}')

    from utils.split_utils import split_indices
    train_idx, _, test_idx = split_indices(len(X), seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cols = [f'f{i}' for i in range(X.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test, columns=cols)
    test_df['label'] = y_test

    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(label='label', eval_metric='roc_auc')
    predictor.fit(train_data=train_df, time_limit=args.time_limit, verbosity=2)
    results = predictor.evaluate(test_df, silent=True)
    print(f'AutoGluon MSLR-cls: {results}')


if __name__ == '__main__':
    main()
