import torch
from torch.utils.data import Dataset
import pandas as pd


class ReconstructDataset(Dataset):
    def __init__(self, df, cont_columns, cat_columns):
        """
        Dataset for reconstruction task.

        :param df: pandas.DataFrame, complete data.
        :param cont_columns: continuous feature column names (will be standardized)
        :param cat_columns: categorical feature column names (keep integer encoding)
        """
        self.df = df
        self.cont_columns = cont_columns
        self.cat_columns = cat_columns
        
        # Keep original column order
        self.all_columns = list(df.columns)  # Use DataFrame's original column order
        self.column_types = {col: 'cont' if col in cont_columns else 'cat' for col in self.all_columns}
        
        # Convert all features to tensors
        self.values = {}
        for col in self.all_columns:
            if col in cont_columns:
                self.values[col] = torch.tensor(df[col].values, dtype=torch.float32)
            else:
                self.values[col] = torch.tensor(df[col].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x: feature tensor arranged in original order
            y: target tensor arranged in original order
        """
        # Get features in original order
        x = torch.stack([self.values[col][idx] for col in self.all_columns])
        y = x.clone()  # Clone to prevent inplace modification
        
        return x, y


class IndexedReconstructDataset(Dataset):
    def __init__(self, df, cont_indices, cat_indices):
        """
        Dataset for reconstruction task (index-based version).

        :param df: pandas.DataFrame, complete data.
        :param cont_indices: list of continuous feature indices
        :param cat_indices: list of categorical feature indices
        """
        self.df = df
        self.cont_indices = cont_indices
        self.cat_indices = cat_indices
        
        # Keep original column order
        self.all_columns = list(df.columns)  # Use DataFrame's original column order
        self.column_types = {}
        
        # Determine type of each column based on indices
        for i, col in enumerate(self.all_columns):
            if i in cont_indices:
                self.column_types[col] = 'cont'
            elif i in cat_indices:
                self.column_types[col] = 'cat'
            else:
                # By default, if index not in any list, consider as continuous feature
                self.column_types[col] = 'cont'
        
        # Convert all features to tensors
        self.values = {}
        for i, col in enumerate(self.all_columns):
            if i in cont_indices:
                self.values[col] = torch.tensor(df[col].values, dtype=torch.float32)
            else:
                self.values[col] = torch.tensor(df[col].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x: feature tensor arranged in original order
            y: target tensor arranged in original order
        """
        # Get features in original order
        x = torch.stack([self.values[col][idx] for col in self.all_columns])
        y = x.clone()  # Clone to prevent inplace modification
        
        return x, y