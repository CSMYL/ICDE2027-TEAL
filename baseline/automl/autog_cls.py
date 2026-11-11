import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor
import time

data_dir = 'cls'
seeds = [42, 142, 127, 723, 2025]

all_results = []

for filename in os.listdir(data_dir):
    
    start_time = time.time()
    if not filename.endswith('.csv'):
        continue

    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)

    target_column = df.columns[-1]
    print(f"\n===== Processing {filename} =====")

    auc_list = []

    for i, seed in enumerate(seeds):
        try:
            train_val_df, test_df = train_test_split(
                df, test_size=0.2, stratify=df[target_column], random_state=seed
            )

            train_df, val_df = train_test_split(
                train_val_df, test_size=0.2, stratify=train_val_df[target_column], random_state=seed
            )

            train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

            predictor = TabularPredictor(label=target_column, eval_metric='roc_auc').fit(
                train_data=train_full_df,
                tuning_data=val_df,
                time_limit=480,
                verbosity=0
            )

            y_true = test_df[target_column]
            y_pred_proba = predictor.predict_proba(test_df)
            num_classes = len(y_true.unique())

            if num_classes <= 2:
                auc = roc_auc_score(y_true, y_pred_proba.iloc[:, 1])
            else:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')

            print(f"  [Seed {seed}] AUC: {auc:.4f}")
            auc_list.append(auc)

        except Exception as e:
            print(f"  [Seed {seed}] AUC calculation failed: {e}")
            auc_list.append(float('nan'))

    auc_series = pd.Series(auc_list)
    elapsed_time = time.time() - start_time
    
    result_record = {
        'dataset': filename,
        'auc_1': auc_list[0],
        'auc_2': auc_list[1],
        'auc_3': auc_list[2],
        'auc_4': auc_list[3],
        'auc_5': auc_list[4],
        'auc_mean': auc_series.mean(),
        'auc_std': auc_series.std(),
        'time_seconds': elapsed_time
    }
    all_results.append(result_record)

results_df = pd.DataFrame(all_results)
results_df.to_csv('cls_results.csv', index=False)
print("\nAll results saved to 'cls_results.csv'")