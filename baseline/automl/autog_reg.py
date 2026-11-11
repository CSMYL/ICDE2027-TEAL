import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from autogluon.tabular import TabularPredictor

data_dir = 'reg'
seeds = [42, 142, 127, 723, 2025]

all_results = []

for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'):
        continue

    start_time = time.time()
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)

    target_column = df.columns[-1]
    print(f"\n===== Processing {filename} =====")

    rmse_list = []

    for seed in seeds:
        try:
            train_val_df, test_df = train_test_split(
                df, test_size=0.2, random_state=seed
            )

            train_df, val_df = train_test_split(
                train_val_df, test_size=0.2, random_state=seed
            )

            train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

            predictor = TabularPredictor(label=target_column, problem_type='regression', eval_metric='rmse').fit(
                train_data=train_full_df,
                tuning_data=val_df,
                time_limit=480,
                verbosity=0
            )

            y_true = test_df[target_column]
            y_pred = predictor.predict(test_df)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            print(f"  [Seed {seed}] RMSE: {rmse:.4f}")
            rmse_list.append(rmse)

        except Exception as e:
            print(f"  [Seed {seed}] RMSE calculation failed: {e}")
            rmse_list.append(float('nan'))

    elapsed_time = time.time() - start_time

    rmse_series = pd.Series(rmse_list)
    result_record = {
        'dataset': filename,
        'rmse_1': rmse_list[0],
        'rmse_2': rmse_list[1],
        'rmse_3': rmse_list[2],
        'rmse_4': rmse_list[3],
        'rmse_5': rmse_list[4],
        'rmse_mean': rmse_series.mean(),
        'rmse_std': rmse_series.std(),
        'time_seconds': elapsed_time
    }
    all_results.append(result_record)

results_df = pd.DataFrame(all_results)
results_df.to_csv('regression_results.csv', index=False)
print("\nAll results saved to 'regression_results.csv'")
