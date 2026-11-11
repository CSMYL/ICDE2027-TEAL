import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset
from datasets.adult_dataset import load_adult_reconstruct_dataset
from datasets.cardio_dataset import load_cardio_reconstruct_dataset
from datasets.diamonds_dataset import load_diamonds_reconstruct_dataset
from datasets.elevator_dataset import load_elevator_reconstruct_dataset
from datasets.housesale_dataset import load_housesale_reconstruct_dataset
from datasets.crime_dataset import load_crime_reconstruct_dataset
from datasets.meps_dataset import load_meps_reconstruct_dataset

class BaselineDataset(Dataset):
    """Baseline-specific dataset, standard format with input x and output y"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_creditcard_baseline_dataset():
    """Load CreditCard dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_creditcard_reconstruct_dataset()
    
    sample_x, sample_y = dataset_obj[0]
    
    target_idx = len(v) - 1
    
    x_data = []
    y_data = []
    
    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)
        
        x_data.append(x)
        y_data.append(y)
    
    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)
    
    baseline_dataset = BaselineDataset(x_tensor, y_tensor)
    
    return baseline_dataset, v, num_classes_dict, target_idx


def load_adult_baseline_dataset():
    """Load Adult dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_adult_reconstruct_dataset()
    
    target_idx = len(v) - 1
    
    x_data = []
    y_data = []
    
    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)
        
        x_data.append(x)
        y_data.append(y)
    
    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)
    
    baseline_dataset = BaselineDataset(x_tensor, y_tensor)
    
    return baseline_dataset, v, num_classes_dict, target_idx

def load_cardio_baseline_dataset():
    """Load Cardio dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_cardio_reconstruct_dataset()
    
    target_idx = len(v) - 1
    
    x_data = []
    y_data = []
    
    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)
        
        x_data.append(x)
        y_data.append(y)
    
    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)
    
    baseline_dataset = BaselineDataset(x_tensor, y_tensor)
    
    return baseline_dataset, v, num_classes_dict, target_idx

def load_diamonds_baseline_dataset():
    """Load Diamonds dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_diamonds_reconstruct_dataset()
    
    target_idx = len(v) - 1
    
    x_data = []
    y_data = []
    
    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)
        
        x_data.append(x)
        y_data.append(y)
    
    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)
    
    baseline_dataset = BaselineDataset(x_tensor, y_tensor)
    
    return baseline_dataset, v, num_classes_dict, target_idx

def load_elevator_baseline_dataset():
    dataset_obj, v, num_classes_dict = load_elevator_reconstruct_dataset()
    target_idx = len(v) - 1
    x_data = []
    y_data = []
    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)
        x_data.append(x)
        y_data.append(y)
    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)
    baseline_dataset = BaselineDataset(x_tensor, y_tensor)
    return baseline_dataset, v, num_classes_dict, target_idx

def load_housesale_baseline_dataset():
    """Load Housesale dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_housesale_reconstruct_dataset()
    target_idx = len(v) - 1

    x_data = []
    y_data = []

    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)

        x_data.append(x)
        y_data.append(y)

    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)

    baseline_dataset = BaselineDataset(x_tensor, y_tensor)

    return baseline_dataset, v, num_classes_dict, target_idx

def load_crime_baseline_dataset():
    """Load Crime dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_crime_reconstruct_dataset()
    target_idx = len(v) - 1

    x_data = []
    y_data = []

    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)

        x_data.append(x)
        y_data.append(y)

    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)

    baseline_dataset = BaselineDataset(x_tensor, y_tensor)

    return baseline_dataset, v, num_classes_dict, target_idx

def load_meps_baseline_dataset():
    """Load MEPS dataset, convert to baseline format"""
    dataset_obj, v, num_classes_dict = load_meps_reconstruct_dataset()
    target_idx = len(v) - 1

    x_data = []
    y_data = []

    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)

        x_data.append(x)
        y_data.append(y)

    x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)

    baseline_dataset = BaselineDataset(x_tensor, y_tensor)

    return baseline_dataset, v, num_classes_dict, target_idx