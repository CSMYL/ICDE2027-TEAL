import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

def run_autogluon_regression(data_path, time_limit=480, seeds=[42, 142, 127, 723, 2025]):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("AutoGluon not available, skipping...")
        return None
    
    df = pd.read_csv(data_path)
    target_column = df.columns[-1]
    
    rmse_list = []
    
    for seed in seeds:
        try:
            from utils.split_utils import split_indices
            train_idx, val_idx, test_idx = split_indices(len(df), seed=seed)
            test_df = df.iloc[test_idx.tolist()]
            train_df = df.iloc[train_idx.tolist()]
            val_df = df.iloc[val_idx.tolist()]
            train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

            predictor = TabularPredictor(
                label=target_column, 
                problem_type='regression', 
                eval_metric='rmse'
            ).fit(
                train_data=train_full_df,
                tuning_data=val_df,
                time_limit=time_limit,
                verbosity=0
            )

            y_true = test_df[target_column]
            y_pred = predictor.predict(test_df)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            rmse_list.append(rmse)
            print(f"  [AutoGluon Seed {seed}] RMSE: {rmse:.4f}")

        except Exception as e:
            print(f"  [AutoGluon Seed {seed}] Failed: {e}")
            rmse_list.append(float('nan'))
    
    return pd.Series(rmse_list).mean() if rmse_list else None

def run_autosklearn_regression(data_path, time_limit=300, seeds=[42, 142, 127, 723, 2025]):
    try:
        from autosklearn.regression import AutoSklearnRegressor
    except ImportError:
        print("AutoSklearn not available, skipping...")
        return None
    
    df = pd.read_csv(data_path)
    target_column = df.columns[-1]
    
    rmse_list = []
    
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
            
            model = AutoSklearnRegressor(
                time_left_for_this_task=time_limit, 
                per_run_time_limit=60, 
                seed=seed
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_list.append(rmse)
            print(f"  [AutoSklearn Seed {seed}] RMSE: {rmse:.4f}")
            
        except Exception as e:
            print(f"  [AutoSklearn Seed {seed}] Failed: {e}")
            rmse_list.append(float('nan'))
    
    return pd.Series(rmse_list).mean() if rmse_list else None

def main():
    parser = argparse.ArgumentParser(description='AutoML Regression Benchmark')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='reg', help='Data directory')
    parser.add_argument('--time_limit', type=int, default=300, help='Time limit per run in seconds')
    
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
    if not os.path.exists(data_path):
        print(f"Dataset file {data_path} not found!")
        return
    
    print(f"===== Processing {args.dataset} =====")
    
    print("Running AutoGluon...")
    start_time = time.time()
    ag_rmse = run_autogluon_regression(data_path, args.time_limit)
    ag_time = time.time() - start_time
    print(f"AutoGluon - Mean RMSE: {ag_rmse:.4f}, Time: {ag_time:.2f}s")

    print("Running AutoSklearn...")
    start_time = time.time()
    askl_rmse = run_autosklearn_regression(data_path, args.time_limit)
    askl_time = time.time() - start_time
    print(f"AutoSklearn - Mean RMSE: {askl_rmse:.4f}, Time: {askl_time:.2f}s")

if __name__ == "__main__":
    main()