import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_datasets import (
    load_creditcard_baseline_dataset,
    load_adult_baseline_dataset,
    load_cardio_baseline_dataset,
    load_diamonds_baseline_dataset,
    load_elevator_baseline_dataset,
    load_housesale_baseline_dataset,
    load_crime_baseline_dataset,
    load_meps_baseline_dataset
)

def calculate_causal_effects(data, target_col, feature_cols):
    """
    Calculate causal effects using DoWhy
    Args:
        data: DataFrame containing all features and target
        target_col: target column name
        feature_cols: list of feature column names
    Returns:
        causal_strengths: causal strength (absolute value) for each feature
    """
    try:
        import dowhy
        from dowhy import CausalModel
        import networkx as nx
        
        causal_strengths = []
        
        for feature in feature_cols:
            edges = [(feature, target_col)]
            causal_graph = nx.DiGraph(edges)
            
            model = CausalModel(
                data=data,
                treatment=[feature],
                outcome=target_col,
                graph=causal_graph
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            
            causal_strength = np.abs(estimate.value)
            causal_strengths.append(causal_strength)
        
        return np.array(causal_strengths)
        
    except ImportError:
        print("DoWhy library not installed, using uniform regularization")
        return np.ones(len(feature_cols))
    except Exception as e:
        print(f"Causal effect calculation failed: {e}, using uniform regularization")
        return np.ones(len(feature_cols))

def adaptive_l1_loss(weights, causal_strengths, lambda_l1=0.01, min_strength=0.1, max_strength=10.0):
    """
    Adaptive L1 regularization based on causal strength
    Args:
        weights: model weights [out_features, in_features]
        causal_strengths: causal strength for each feature
        lambda_l1: base L1 regularization coefficient
        min_strength: minimum regularization strength
        max_strength: maximum regularization strength
    Returns:
        L1 regularization loss value
    """
    if not isinstance(causal_strengths, np.ndarray):
        causal_strengths = np.array(causal_strengths)
    
    if len(causal_strengths) > 0:
        normalized_strengths = (causal_strengths - causal_strengths.min()) / (causal_strengths.max() - causal_strengths.min() + 1e-8)
        regularization_strengths = 1.0 - normalized_strengths
        regularization_strengths = min_strength + (max_strength - min_strength) * regularization_strengths
    else:
        regularization_strengths = np.ones(weights.shape[1])
    
    l1_norm = torch.sum(torch.abs(weights) * torch.tensor(regularization_strengths, dtype=weights.dtype, device=weights.device))
    return lambda_l1 * l1_norm

def convert_dataset_to_numpy(dataset, v, target_idx):
    all_data = []
    all_targets = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_data.append(x.numpy())
        all_targets.append(y.item())
    data_array = np.array(all_data)
    target_array = np.array(all_targets)
    return data_array, target_array

class LinearDataset(Dataset):
    def __init__(self, data, targets, target_idx):
        self.data = data
        self.targets = targets
        self.target_idx = target_idx
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        x = self.data[idx]
        input_features = np.concatenate([x[:self.target_idx], x[self.target_idx+1:]])
        y = self.targets[idx]
        return torch.tensor(input_features, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LinearBaselineModel(nn.Module):
    """Linear regression/Logistic regression model"""
    def __init__(self, input_dim, is_regression=True):
        super().__init__()
        self.input_dim = input_dim
        self.is_regression = is_regression
        
        # Linear layer
        self.linear = nn.Linear(input_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def custom_l1_loss(weights, lambda_l1=0.01):
    """
    Custom L1 regularization loss
    Args:
        weights: model weights
        lambda_l1: L1 regularization coefficient
    Returns:
        L1 regularization loss value
    """
    l1_norm = torch.sum(torch.abs(weights))
    return lambda_l1 * l1_norm

def test_linear_baseline(dataset="creditcard", gpu_id=-1, 
                        batch_size=64, val_batch_size=64, epochs=20, lr=1e-3,
                        num_workers=4, lambda_l1=0.01):
    torch.manual_seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Parameter settings: dataset={dataset}, batch_size={batch_size}, epochs={epochs}, lr={lr}, lambda_l1={lambda_l1}")

    # Data loading
    if dataset.lower() == "creditcard":
        dataset_obj, v, num_classes_dict, target_idx = load_creditcard_baseline_dataset()
        dataset_name = "CreditCard"
    elif dataset.lower() == "adult":
        dataset_obj, v, num_classes_dict, target_idx = load_adult_baseline_dataset()
        dataset_name = "Adult"
    elif dataset.lower() == "cardio":
        dataset_obj, v, num_classes_dict, target_idx = load_cardio_baseline_dataset()
        dataset_name = "Cardio"
    elif dataset.lower() == "diamonds":
        dataset_obj, v, num_classes_dict, target_idx = load_diamonds_baseline_dataset()
        dataset_name = "Diamonds"
    elif dataset.lower() == "elevator":
        dataset_obj, v, num_classes_dict, target_idx = load_elevator_baseline_dataset()
        dataset_name = "Elevator"
    elif dataset.lower() == "housesale":
        dataset_obj, v, num_classes_dict, target_idx = load_housesale_baseline_dataset()
        dataset_name = "Housesale"
    elif dataset.lower() == "crime":
        dataset_obj, v, num_classes_dict, target_idx = load_crime_baseline_dataset()
        dataset_name = "Crime"
    elif dataset.lower() == "meps":
        dataset_obj, v, num_classes_dict, target_idx = load_meps_baseline_dataset()
        dataset_name = "MEPS"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)

    logger.info(f"Dataset: {dataset_name}, size: {len(dataset_obj)}")
    logger.info(f"Feature type: {v}, target index: {target_idx}")

    from utils.split_utils import split_indices
    total_size = len(dataset_obj)
    train_idx, val_idx, test_idx = split_indices(total_size, seed=42)
    train_dataset = torch.utils.data.Subset(dataset_obj, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(dataset_obj, val_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset_obj, test_idx.tolist())
    logger.info(f"Train set: {len(train_dataset)}, Validation set: {len(val_dataset)}, Test set: {len(test_dataset)}")

    # Determine task type
    is_regression = v[target_idx] == 0
    task_type = "regression" if is_regression else "classification"
    logger.info(f"Task type: {task_type}")

    # Convert to numpy
    train_data, train_targets = convert_dataset_to_numpy(train_dataset, v, target_idx)
    val_data, val_targets = convert_dataset_to_numpy(val_dataset, v, target_idx)
    test_data, test_targets = convert_dataset_to_numpy(test_dataset, v, target_idx)

    # Calculate causal effects
    print("Calculating causal effects...")
    # Create DataFrame for DoWhy
    feature_cols = [f"feature_{i}" for i in range(len(v)) if i != target_idx]
    target_col = f"feature_{target_idx}"
    
    # Concatenate training data
    all_data = np.concatenate([train_data, val_data, test_data], axis=0)
    all_targets = np.concatenate([train_targets, val_targets, test_targets], axis=0)
    
    # Create full dataset
    df = pd.DataFrame(all_data, columns=[f"feature_{i}" for i in range(len(v))])
    df[target_col] = all_targets
    
    # Calculate causal effects
    causal_strengths = calculate_causal_effects(df, target_col, feature_cols)
    print(f"Causal effect strength vector: {causal_strengths}")
    print(f"Causal effect strength range: [{causal_strengths.min():.6f}, {causal_strengths.max():.6f}]")
    print(f"Causal effect strength mean: {causal_strengths.mean():.6f}")
    print(f"Causal effect strength standard deviation: {causal_strengths.std():.6f}")
    
    # Display causal effects for each feature
    print("Causal effect weights for each feature:")
    for i, strength in enumerate(causal_strengths):
        print(f"  Feature{i}: {strength:.6f}")

    train_ds = LinearDataset(train_data, train_targets, target_idx)
    val_ds = LinearDataset(val_data, val_targets, target_idx)
    test_ds = LinearDataset(test_data, test_targets, target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("GPU not available, using CPU")

    # Calculate input feature dimension (excluding target feature)
    input_dim = len(v) - 1  # Total features minus target feature
    logger.info(f"Input feature dimension: {input_dim}")

    # Model
    model = LinearBaselineModel(input_dim, is_regression).to(device)

    # Calculate and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Linear model total parameters: {total_params:,}")
    logger.info(f"Linear trainable parameters: {trainable_params:,}")
    logger.info(f"Linear model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # Select loss function based on task type
    if is_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def evaluate(loader):
        model.eval()
        all_labels, all_predictions = [], []
        infer_start = time.time()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                all_labels.append(y.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
        infer_end = time.time()
        infer_time = infer_end - infer_start
        num_samples = len(loader.dataset)
        logger.info(f"Inference total time: {infer_time:.2f} seconds, average per sample: {infer_time/num_samples:.4f} seconds")
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        
        if is_regression:
            # Regression task: calculate MSE and R2
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(all_labels, all_predictions)
            r2 = r2_score(all_labels, all_predictions)
            return mse, r2
        else:
            # Classification task: calculate accuracy and AUC
            probs = torch.sigmoid(torch.tensor(all_predictions)).numpy()
            preds = (probs > 0.5).astype(int)
            acc = accuracy_score(all_labels, preds)
            auc = roc_auc_score(all_labels, probs)
            return acc, auc

    best_val_metric = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            predictions = model(x)
            loss = criterion(predictions, y)
            
            # Add Caual L1 regularization
            l1_loss = adaptive_l1_loss(model.linear.weight, causal_strengths, lambda_l1)
            total_loss = loss + l1_loss
            
            total_loss.backward()
            optimizer.step()
        
        val_metric1, val_metric2 = evaluate(val_loader)
        
        # End of training loop
        epoch_end = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} time: {epoch_end - epoch_start:.2f} seconds")
        
        if is_regression:
            print(f"Epoch {epoch}: val_mse={val_metric1:.4f}, val_r2={val_metric2:.4f}")
            if val_metric2 > best_val_metric:
                best_val_metric = val_metric2
        else:
            print(f"Epoch {epoch}: val_acc={val_metric1:.4f}, val_auc={val_metric2:.4f}")
            if val_metric2 > best_val_metric:
                best_val_metric = val_metric2

    # Test set evaluation
    test_metric1, test_metric2 = evaluate(test_loader)
    
    if is_regression:
        print(f"Test MSE: {test_metric1:.4f}, Test R2: {test_metric2:.4f}")
    else:
        print(f"Test acc: {test_metric1:.4f}, Test AUC: {test_metric2:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Linear Regression/Logistic Regression Baseline')
    parser.add_argument('--dataset', type=str, default='housesale',
                       choices=['creditcard', 'adult', 'cardio', 'diamonds', 'elevator', 'housesale', 'crime', 'meps'],
                       help='Dataset selection')
    parser.add_argument('--gpu_id', type=int, default=-1, 
                       help='GPU ID, -1 for automatic selection')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=512,
                       help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--lambda_l1', type=float, default=5e-2,
                       help='L1 regularization coefficient')
    
    args = parser.parse_args()
    
    # Update function call, pass all parameters
    test_linear_baseline(
        dataset=args.dataset,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        lambda_l1=args.lambda_l1
    ) 