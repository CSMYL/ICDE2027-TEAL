import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # CAIM_code/
sys.path.insert(0, project_root)

from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset
from datasets.adult_dataset import load_adult_reconstruct_dataset
from datasets.cardio_dataset import load_cardio_reconstruct_dataset
from datasets.diamonds_dataset import load_diamonds_reconstruct_dataset
from datasets.elevator_dataset import load_elevator_reconstruct_dataset
from datasets.housesale_dataset import load_housesale_reconstruct_dataset
from datasets.crime_dataset import load_crime_reconstruct_dataset
from datasets.meps_dataset import load_meps_reconstruct_dataset

from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """Simple x→y dataset wrapper for baseline evaluation."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _make_baseline_dataset(load_fn):
    """Generic loader: calls load_fn(), extracts data, builds BaselineDataset."""
    dataset_obj, v, num_classes_dict = load_fn()
    target_idx = len(v) - 1

    x_list, y_list = [], []
    for i in range(len(dataset_obj)):
        sample_x, sample_y = dataset_obj[i]
        x = sample_x.clone()
        x[target_idx] = 0
        y = sample_y[target_idx].unsqueeze(0)
        x_list.append(x)
        y_list.append(y)

    x_tensor = torch.stack(x_list)
    y_tensor = torch.stack(y_list)
    return BaselineDataset(x_tensor, y_tensor), v, num_classes_dict, target_idx


# ---------------------------------------------------------------------------
# Public loaders — one per dataset
# ---------------------------------------------------------------------------

def load_creditcard_baseline_dataset():
    return _make_baseline_dataset(load_creditcard_reconstruct_dataset)


def load_adult_baseline_dataset():
    return _make_baseline_dataset(load_adult_reconstruct_dataset)


def load_cardio_baseline_dataset():
    return _make_baseline_dataset(load_cardio_reconstruct_dataset)


def load_diamonds_baseline_dataset():
    return _make_baseline_dataset(load_diamonds_reconstruct_dataset)




def load_elevator_baseline_dataset():
    return _make_baseline_dataset(load_elevator_reconstruct_dataset)


def load_housesale_baseline_dataset():
    return _make_baseline_dataset(load_housesale_reconstruct_dataset)


def load_crime_baseline_dataset():
    return _make_baseline_dataset(load_crime_reconstruct_dataset)


def load_meps_baseline_dataset():
    return _make_baseline_dataset(load_meps_reconstruct_dataset)
