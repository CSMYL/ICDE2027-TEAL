import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

def run_autogluon_classification(data_path, time_limit=480, seeds=[42, 142, 127, 723, 2025]):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("AutoGluon not available, skipping...")
        return None
    
    df = pd.read_csv(data_path)
    target_column = df.columns[-1]
    
    auc_list = []
    
    for seed in seeds:
        try:
            from utils.split_utils import split_indices
            train_idx, val_idx, test_idx = split_indices(len(df), seed=seed)
            test_df = df.iloc[test_idx.tolist()]
            train_df = df.iloc[train_idx.tolist()]
            val_df = df.iloc[val_idx.tolist()]
            train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

            predictor = TabularPredictor(label=target_column, eval_metric='roc_auc').fit(
                train_data=train_full_df,
                tuning_data=val_df,
                time_limit=time_limit,
                verbosity=0
            )

            y_true = test_df[target_column]
            y_pred_proba = predictor.predict_proba(test_df)
            num_classes = len(y_true.unique())

            if num_classes <= 2:
                auc = roc_auc_score(y_true, y_pred_proba.iloc[:, 1])
            else:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')

            auc_list.append(auc)
            print(f"  [AutoGluon Seed {seed}] AUC: {auc:.4f}")

        except Exception as e:
            print(f"  [AutoGluon Seed {seed}] Failed: {e}")
            auc_list.append(float('nan'))
    
    return pd.Series(auc_list).mean() if auc_list else None

def run_autosklearn_classification(data_path, time_limit=300, seeds=[42, 142, 127, 723, 2025]):
    try:
        from autosklearn.classification import AutoSklearnClassifier
    except ImportError:
        print("AutoSklearn not available, skipping...")
        return None
    
    df = pd.read_csv(data_path)
    target_column = df.columns[-1]
    
    auc_list = []
    
    for seed in seeds:
        try:
            from utils.split_utils import split_indices
            train_idx, val_idx, test_idx = split_indices(len(df), seed=seed)
            test_df = df.iloc[test_idx.tolist()]
            train_df = df.iloc[train_idx.tolist()]
            val_df = df.iloc[val_idx.tolist()]
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            model = AutoSklearnClassifier(
                time_left_for_this_task=time_limit, 
                per_run_time_limit=60, 
                seed=seed
            )
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba)
            auc_list.append(auc)
            print(f"  [AutoSklearn Seed {seed}] AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"  [AutoSklearn Seed {seed}] Failed: {e}")
            auc_list.append(float('nan'))
    
    return pd.Series(auc_list).mean() if auc_list else None

def main():
    parser = argparse.ArgumentParser(description='AutoML Classification Benchmark')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='cls', help='Data directory')
    parser.add_argument('--time_limit', type=int, default=300, help='Time limit per run in seconds')
    
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
    if not os.path.exists(data_path):
        print(f"Dataset file {data_path} not found!")
        return
    
    print(f"===== Processing {args.dataset} =====")
    
    print("Running AutoGluon...")
    start_time = time.time()
    ag_auc = run_autogluon_classification(data_path, args.time_limit)
    ag_time = time.time() - start_time
    print(f"AutoGluon - Mean AUC: {ag_auc:.4f}, Time: {ag_time:.2f}s")
    
    print("Running AutoSklearn...")
    start_time = time.time()
    askl_auc = run_autosklearn_classification(data_path, args.time_limit)
    askl_time = time.time() - start_time
    print(f"AutoSklearn - Mean AUC: {askl_auc:.4f}, Time: {askl_time:.2f}s")

if __name__ == "__main__":
    main()