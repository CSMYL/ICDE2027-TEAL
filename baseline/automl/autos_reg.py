import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from autosklearn.regression import AutoSklearnRegressor

data_dir = 'reg'
seeds = [42, 142, 127, 723, 2025]
all_results = []

for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'): continue
    
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    target_column = df.columns[-1]
    print(f"Processing {filename}")
    
    rmse_list = []
    for seed in seeds:
        try:
            train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
            train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=seed)
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            model = AutoSklearnRegressor(time_left_for_this_task=300, per_run_time_limit=60, seed=seed)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_list.append(rmse)
            print(f"  Seed {seed}: RMSE = {rmse:.4f}")
            
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")
            rmse_list.append(float('nan'))
    
    rmse_series = pd.Series(rmse_list)
    all_results.append({
        'dataset': filename,
        'rmse_mean': rmse_series.mean(),
        'rmse_std': rmse_series.std(),
        **{f'rmse_{i+1}': rmse for i, rmse in enumerate(rmse_list)}
    })

pd.DataFrame(all_results).to_csv('sk_reg_results.csv', index=False)