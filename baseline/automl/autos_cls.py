import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from autosklearn.classification import AutoSklearnClassifier

data_dir = 'cls'
seeds = [42, 142, 127, 723, 2025]
all_results = []

for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'): continue
    
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    target_column = df.columns[-1]
    print(f"Processing {filename}")
    
    auc_list = []
    for seed in seeds:
        try:
            train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df[target_column], random_state=seed)
            train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[target_column], random_state=seed)
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            model = AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=60, seed=seed)
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba)
            auc_list.append(auc)
            print(f"  Seed {seed}: AUC = {auc:.4f}")
            
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")
            auc_list.append(float('nan'))
    
    auc_series = pd.Series(auc_list)
    all_results.append({
        'dataset': filename,
        'auc_mean': auc_series.mean(),
        'auc_std': auc_series.std(),
        **{f'auc_{i+1}': auc for i, auc in enumerate(auc_list)}
    })

pd.DataFrame(all_results).to_csv('sk_cls_results.csv', index=False)