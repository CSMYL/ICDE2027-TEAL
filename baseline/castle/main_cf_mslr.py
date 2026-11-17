import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.mslr_loader import load_mslr_cls
from CASTLE_CF import CASTLE


def one_hot_encode(df, target_col='y'):
    """One-hot encode the target column, appending one-hot columns to the end."""
    encoder = OneHotEncoder(sparse=False, dtype=int)
    y_reshaped = df[[target_col]].values
    y_encoded = encoder.fit_transform(y_reshaped)
    class_names = encoder.categories_[0]
    onehot_columns = [f'y{int(cls)}' for cls in class_names]
    y_encoded_df = pd.DataFrame(y_encoded, columns=onehot_columns)
    df_encoded = pd.concat([df.drop(target_col, axis=1), y_encoded_df], axis=1)
    return df_encoded, len(onehot_columns)


def normalize_features_only(df, num_y_cols_from_end, center_y=True):
    """Standardize feature columns only, leaving the (one-hot) target untouched."""
    y_cols = df.columns[-num_y_cols_from_end:]
    x_cols = df.columns[:-num_y_cols_from_end]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[x_cols])
    df_x_scaled = pd.DataFrame(X_scaled, columns=x_cols, index=df.index)
    df_y = df[y_cols].copy()
    if center_y:
        df_y -= 0.5
    df_final = pd.concat([df_x_scaled, df_y], axis=1)
    return df_final.values, df[y_cols].values


def main():
    parser = argparse.ArgumentParser(description='CASTLE on MSLR-WEB10K (binary classification)')
    parser.add_argument('--max_samples', type=int, default=10000, help='subsample documents (0 = full ~723k)')
    parser.add_argument('--threshold', type=int, default=0, help='relevance > threshold => positive (default 0)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--reg_lambda', type=float, default=1)
    parser.add_argument('--reg_beta', type=float, default=5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ckpt_file', type=str, default='tmp_mslr.ckpt')
    parser.add_argument('--extension', type=str, default='mslr')
    parser.add_argument('--output_log', type=str, default='castle_mslr.log')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load MSLR-WEB10K as a binary classification task
    X, y = load_mslr_cls(
        max_samples=args.max_samples if args.max_samples > 0 else None,
        binary=True, threshold=args.threshold,
    )
    print(f'MSLR: X {X.shape}, positive rate {y.mean():.3f}')

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    df['y'] = y

    df, num_y_cols_from_end = one_hot_encode(df, target_col='y')

    total_len = len(df)
    dataset_sz = int(total_len * 0.8)
    test_size = total_len - dataset_sz
    df_test = df.iloc[-test_size:]
    df = df.iloc[:dataset_sz]

    df, y_full = normalize_features_only(df, num_y_cols_from_end=num_y_cols_from_end)
    df_test, y_test_full = normalize_features_only(df_test, num_y_cols_from_end=num_y_cols_from_end)

    X_test = df_test
    y_test = y_test_full
    X_DAG = df

    kf = KFold(n_splits=args.n_folds, random_state=1, shuffle=True)

    acc_list = []
    auc_list = []
    fold = 0
    for train_idx, val_idx in kf.split(X_DAG):
        fold += 1
        print(f'fold = {fold}')
        X_train = X_DAG[train_idx]
        y_train = y_full[train_idx]
        X_val = X_DAG[val_idx]
        y_val = y_full[val_idx]

        castle = CASTLE(num_train=X_DAG.shape[0], num_inputs=X_DAG.shape[1],
                        reg_lambda=args.reg_lambda, reg_beta=args.reg_beta,
                        w_threshold=0.3, ckpt_file=args.ckpt_file, y_dim=num_y_cols_from_end)
        num_nodes = np.shape(X_DAG)[1]
        castle.fit(X_train, y_train, num_nodes, X_val, y_val, X_test, y_test)

        y_pred = castle.pred(X_test)
        acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)
        acc_list.append(acc)
        print(f'fold {fold}: Accuracy = {acc:.4f}')

        logits_probs = castle.sess.run(
            tf.nn.softmax(castle.logits),
            feed_dict={castle.X: X_test, castle.keep_prob: 1, castle.is_train: False, castle.noise: 0}
        )
        try:
            auc = roc_auc_score(y_test, logits_probs, multi_class='ovr')
            auc_list.append(auc)
            print(f'fold {fold}: AUC = {auc:.4f}')
        except ValueError as e:
            print('AUC calculation failed:', e)

    print('MEAN Accuracy =', np.mean(acc_list), 'STD =', np.std(acc_list))
    if auc_list:
        print('MEAN AUC =', np.mean(auc_list), 'STD =', np.std(auc_list))


if __name__ == '__main__':
    main()
