import os
import sys
import argparse
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.mslr_loader import load_mslr_cls


def main():
    parser = argparse.ArgumentParser(description='XGBoost on MSLR-WEB10K (binary classification)')
    parser.add_argument('--max_samples', type=int, default=0, help='0 = full train docs (~723k)')
    parser.add_argument('--threshold', type=int, default=0,
                        help='relevance > threshold => positive (default 0)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X, y = load_mslr_cls(
        max_samples=args.max_samples if args.max_samples > 0 else None,
        binary=True, threshold=args.threshold,
    )
    print(f'MSLR XGBoost-cls: X {X.shape}, y {y.shape}, positive rate {y.mean():.3f}')

    from utils.split_utils import split_indices
    train_idx, _, test_idx = split_indices(len(X), seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, tree_method='hist', random_state=args.seed,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    from sklearn.metrics import accuracy_score, roc_auc_score
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f'XGBoost MSLR-cls: ACC = {acc:.4f}, AUC = {auc:.4f}, train_time = {train_time:.1f}s')


if __name__ == '__main__':
    main()
