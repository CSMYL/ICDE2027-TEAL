import torch
from torch.utils.data import Dataset
import pandas as pd


class ReconstructDataset(Dataset):
    def __init__(self, df, cont_columns, cat_columns):
        """
        Dataset for reconstruction tasks.

        :param df: pandas.DataFrame, complete data.
        :param cont_columns: list of continuous feature column names (will be standardized)
        :param cat_columns: list of categorical feature column names (kept as integer encoding)
        """
        self.df = df
        self.cont_columns = cont_columns
        self.cat_columns = cat_columns
        self.all_columns = list(df.columns)
        self.column_types = {col: 'cont' if col in cont_columns else 'cat' for col in self.all_columns}
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
            x: feature tensor in original column order
            y: target tensor in original column order
        """
        x = torch.stack([self.values[col][idx] for col in self.all_columns])
        y = x.clone()
        return x, y


class IndexedReconstructDataset(Dataset):
    def __init__(self, df, cont_indices, cat_indices):
        """
        Dataset for reconstruction tasks (index-based version).

        :param df: pandas.DataFrame, complete data.
        :param cont_indices: list of indices for continuous features
        :param cat_indices: list of indices for categorical features
        """
        self.df = df
        self.cont_indices = cont_indices
        self.cat_indices = cat_indices
        self.all_columns = list(df.columns)
        self.column_types = {}
        for i, col in enumerate(self.all_columns):
            if i in cont_indices:
                self.column_types[col] = 'cont'
            elif i in cat_indices:
                self.column_types[col] = 'cat'
            else:
                self.column_types[col] = 'cont'
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
            x: feature tensor in original column order
            y: target tensor in original column order
        """
        x = torch.stack([self.values[col][idx] for col in self.all_columns])
        y = x.clone()
        return x, y