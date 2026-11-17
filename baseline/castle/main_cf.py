import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
# import tensorflow as tf
# Disable TensorFlow 2 behaviour
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os
from sklearn.metrics import mean_squared_error
from CASTLE_CF import CASTLE
from utils import random_dag, gen_data_nonlinear
from signal import signal, SIGINT
from sys import exit
import argparse
from sklearn.metrics import roc_auc_score
import time


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def one_hot_encode(df, target_col='y'):
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    encoder = OneHotEncoder(sparse=False, dtype=int)
    y_reshaped = df[[target_col]].values
    y_encoded = encoder.fit_transform(y_reshaped)
    class_names = encoder.categories_[0]
    onehot_columns = [f'y{int(cls)}' for cls in class_names]
    y_encoded_df = pd.DataFrame(y_encoded, columns=onehot_columns)
    df_encoded = pd.concat([df.drop(target_col, axis=1), y_encoded_df], axis=1)
    return df_encoded, len(onehot_columns)

def normalize_features_only(df, num_y_cols_from_end, center_y=True):
    if num_y_cols_from_end <= 0:
        raise ValueError("num_y_cols_from_end must be positive")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv', type=str, default='creditcard.csv', help='csv file path')

    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument('--num_nodes', type=int, default=10)

    parser.add_argument('--dataset_sz', type=int, default=227844)
    parser.add_argument('--output_log', type=str, default='castle.log')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--reg_lambda', type=float, default=1)
    parser.add_argument('--reg_beta', type=float, default=5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ckpt_file', type=str, default='tmp.ckpt')
    parser.add_argument('--extension', type=str, default='')
    parser.add_argument('--branchf', type=float, default=4)

    args = parser.parse_args()
    signal(SIGINT, handler)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_y_cols_from_end = 0
    if args.random_dag:
        def swap_cols(df, a, b):
            df = df.rename(columns={a: 'temp'})
            df = df.rename(columns={b: a})
            return df.rename(columns={'temp': b})


        def swap_nodes(G, a, b):
            newG = nx.relabel_nodes(G, {a: 'temp'})
            newG = nx.relabel_nodes(newG, {b: a})
            return nx.relabel_nodes(newG, {'temp': b})


        # Random DAG
        num_edges = int(args.num_nodes * args.branchf)
        G = random_dag(args.num_nodes, num_edges)

        noise = random.uniform(0.3, 1.0)
        print("Setting noise to ", noise)

        df = gen_data_nonlinear(G, SIZE=args.dataset_sz, var=noise).iloc[:args.dataset_sz]
        df_test = gen_data_nonlinear(G, SIZE=int(args.dataset_sz * 0.25), var=noise)

        for i in range(len(G.edges())):
            if len(list(G.predecessors(i))) > 0:
                df = swap_cols(df, str(0), str(i))
                df_test = swap_cols(df_test, str(0), str(i))
                G = swap_nodes(G, 0, i)
                break

                # print("Number of parents of G", len(list(G.predecessors(i))))
        print("Edges = ", list(G.edges()))

    else:
        '''
        Toy DAG
        The node '0' is the target in the Toy DAG
        '''
        G = nx.DiGraph()
        for i in range(10):
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 4)
        G.add_edge(2, 5)
        G.add_edge(2, 0)
        G.add_edge(3, 0)
        G.add_edge(3, 6)
        G.add_edge(3, 7)
        G.add_edge(6, 9)
        G.add_edge(0, 8)
        G.add_edge(0, 9)

        if args.csv:
            df = pd.read_csv(args.csv)
            df, num_y_cols_from_end = one_hot_encode(df, target_col='y')

            total_len = len(df)
            # Automatically calculate 80% of data size as training set size
            args.dataset_sz = int(total_len * 0.8)
            test_size = total_len - args.dataset_sz

            df_test = df.iloc[-test_size:]
            df = df.iloc[:args.dataset_sz]
        else:
            df = gen_data_nonlinear(G, SIZE=args.dataset_sz)
            df_test = gen_data_nonlinear(G, SIZE=1000)

    scaler = StandardScaler()
    if args.random_dag:
        df = scaler.fit_transform(df)
    else:
        if args.csv:
            df, y_full = normalize_features_only(df, num_y_cols_from_end=num_y_cols_from_end)

        else:
            df = scaler.fit_transform(df)


    df_test, y_test_full = normalize_features_only(df_test, num_y_cols_from_end=num_y_cols_from_end)

    # Start splitting data here
    X_test = df_test
    y_test = y_test_full
    X_DAG = df

    kf = KFold(n_splits=args.n_folds, random_state=1, shuffle=True)

    fold = 0
    REG_castle = []
    print("Dataset limits are", np.ptp(X_DAG), np.ptp(X_test), np.ptp(y_test))
    for train_idx, val_idx in kf.split(X_DAG):
        fold += 1
        print("fold = ", fold)
        print("******* Doing dataset size = ", args.dataset_sz, "****************")
        X_train = X_DAG[train_idx]
        y_train = y_full[train_idx]
        X_val = X_DAG[val_idx]
        y_val = y_full[val_idx]

        w_threshold = 0.3
        castle = CASTLE(num_train=X_DAG.shape[0], num_inputs=X_DAG.shape[1], reg_lambda=args.reg_lambda,
                        reg_beta=args.reg_beta,
                        w_threshold=w_threshold, ckpt_file=args.ckpt_file, y_dim=num_y_cols_from_end)
        num_nodes = np.shape(X_DAG)[1]
        castle.fit(X_train, y_train, num_nodes, X_val, y_val, X_test, y_test)

        W_est = castle.pred_W(X_DAG)
        print(W_est)

        # Record inference time
        inference_start_time = time.time()
        y_pred = castle.pred(X_test)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        
        # REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
        # print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))
        REG_castle.append(accuracy_score(np.argmax(y_test, axis=1), y_pred))
        print("Accuracy =", accuracy_score(np.argmax(y_test, axis=1), y_pred))

        # Get softmax output of logits
        logits_probs = castle.sess.run(
            tf.nn.softmax(castle.logits),
            feed_dict={castle.X: X_test, castle.keep_prob: 1, castle.is_train: False, castle.noise: 0}
        )

        try:
            auc = roc_auc_score(y_test, logits_probs, multi_class='ovr')
            print("AUC =", auc)
        except ValueError as e:
            print("AUC calculation failed:", e)

        ######################################################################
        ######################################################################
        ######Everything below this point is for logging purposes only.#######
        ######################################################################
        ######################################################################


    def format_str(mean, std):
        return "${0:.6f}".format(round(mean, 6)) + " \pm {0:.6f}$    ".format(round(std, 6))


    with open(args.output_log, "a") as logfile:
        logfile.write(str(args.dataset_sz) + ",  " +
                      format_str(np.mean(REG_castle), np.std(REG_castle)) +
                      '\n')
    print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle))




