import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import torch.nn.functional as F
import time

class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=7, min_delta=0, mask_generator=None, use_prediction_loss=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.mask_generator = mask_generator
        self.use_prediction_loss = use_prediction_loss

    def __call__(self, val_loss, val_prediction_loss=None):
        # Choose loss for early stopping
        if self.use_prediction_loss and val_prediction_loss is not None:
            current_loss = val_prediction_loss
        else:
            current_loss = val_loss
            
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            # Update mask generator when patience accumulates to half
            if self.counter == self.patience // 2 and self.mask_generator is not None:
                self.mask_generator.update_parameters_for_early_stopping()
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0

def get_nodes_at_distance(adj_matrix, target_idx, max_distance=3, threshold=0.45, binary_mask_path=None):
    """
    Calculate nodes at distance 1, 2, 3 from target
    
    Args:
        adj_matrix: [num_features, num_features] adjacency matrix
        target_idx: target node index
        max_distance: maximum distance
        threshold: threshold for binarizing adjacency matrix
        binary_mask_path: file path to store binary_adj, None if not stored
        
    Returns:
        dict: {0: [target], 1: [node list], 2: [node list], 3: [node list]}
    """
    # Process mask matrix with threshold and transpose to get correct directed graph adjacency matrix
    binary_adj = (adj_matrix > threshold).float().t()  # Transpose to get correct direction
    
    # Save binary_adj if storage path specified
    if binary_mask_path is not None:
        import os
        # Ensure directory exists
        os.makedirs(os.path.dirname(binary_mask_path) if os.path.dirname(binary_mask_path) else '.', exist_ok=True)
        
        # Convert binary_adj to numpy array and save
        binary_adj_np = binary_adj.detach().cpu().numpy()
        
        # Write to file in append mode
        with open(binary_mask_path, 'a') as f:
            f.write(f"Binary Adj Matrix (threshold={threshold}):\n")
            f.write(str(binary_adj_np))
            f.write("\n" + "="*50 + "\n")
    
    num_features = adj_matrix.shape[0]
    distances = {i: [] for i in range(max_distance + 1)}  # Include distance 0
    
    # Record visited nodes to avoid duplicate calculation (including target itself)
    visited_nodes = {target_idx}
    
    # Use cumulative method to calculate nodes at each distance
    adj_power = binary_adj  # Initial A^1
    
    for dist in range(1, max_distance + 1):
        if dist == 1:
            # Distance 1 nodes: direct neighbors
            dist_candidates = adj_power[:, target_idx].nonzero(as_tuple=True)[0]
        else:
            # Cumulative calculation A^dist
            adj_power = torch.mm(adj_power, binary_adj)
            
            # Get reachable nodes
            dist_candidates = adj_power[:, target_idx].nonzero(as_tuple=True)[0]
        
        # Exclude visited nodes (operate on GPU)
        dist_nodes = []
        for node in dist_candidates:
            if node.item() not in visited_nodes:
                dist_nodes.append(node.item())
                visited_nodes.add(node.item())
        
        distances[dist] = dist_nodes
    
    # Set target's distance to 0
    distances[0] = [target_idx]
    
    return distances

def notears_dag_loss(adj_matrix):
    """Calculate NOTEARS DAG constraint loss, using CASTLE's method
    
    Args:
        adj_matrix: adjacency matrix, shape [num_features, num_features]
        
    Returns:
        torch.Tensor: DAG constraint loss
    """
    d = adj_matrix.shape[0]
    # Square the weight matrix
    Z = adj_matrix * adj_matrix
    
    # Use truncated power series expansion
    dag_l = torch.tensor(d, dtype=torch.float32, device=adj_matrix.device)
    Z_in = torch.eye(d, device=adj_matrix.device)
    coff = 1.0
    
    for i in range(1, 10):
        Z_in = torch.matmul(Z_in, Z)
        dag_l = dag_l + (1.0 / coff) * torch.trace(Z_in)
        coff = coff * (i + 1)
    
    # Final DAG constraint
    h = dag_l - d
    
    return h

def compute_prediction_accuracy(model, data_loader, target_idx, device, v=None):
    """
    Compute prediction accuracy for evaluation during validation
    
    Args:
        model: model
        data_loader: data loader
        target_idx: target feature index
        device: device
        v: feature type vector, used to determine target feature type
    
    Returns:
        float: accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Get prediction results - using the same way as the test set
            outputs, prediction, mask = model(data, target_idx=target_idx, task_type='prediction')
            pred = prediction.squeeze(-1)
            
            # Get true value of target feature
            target_values = target[:, target_idx]
            
            # Use the same prediction processing method as the test set
            if v is not None and target_idx < len(v):
                target_type = v[target_idx]
                if target_type == 0:  # Numerical feature
                    # Regression task, directly use prediction value
                    pred_classes = pred
                else:  # Categorical feature
                    # Classification task
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        # Multi-class case
                        pred_classes = pred.argmax(dim=-1)
                    else:
                        # Binary case
                        pred_probs = torch.sigmoid(pred) if pred.size(-1) == 1 else pred
                        pred_classes = (pred_probs > 0.5).long().view(-1)
            else:
                # Default to classification (backward compatibility)
                if pred.dim() > 1 and pred.size(-1) > 1:
                    pred_classes = pred.argmax(dim=-1)
                else:
                    pred_probs = torch.sigmoid(pred) if pred.size(-1) == 1 else pred
                    pred_classes = (pred_probs > 0.5).long().view(-1)
            
            correct += (pred_classes == target_values).sum().item()
            total += target_values.size(0)
    
    return correct / total if total > 0 else 0.0

def compute_auc_score(model, data_loader, target_idx, device, v=None):
    """
    Compute AUC score
    
    Args:
        model: model
        data_loader: data loader
        target_idx: target feature index
        device: device
        v: feature type vector, used to determine target feature type
    
    Returns:
        float: AUC score
    """
    model.eval()
    all_targets = []
    all_pred_probs = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Get prediction results
            outputs, prediction, mask = model(data, target_idx=target_idx, task_type='prediction')
            pred = prediction.squeeze(-1)
            
            # Process prediction results based on target feature type
            if v is not None and target_idx < len(v):
                target_type = v[target_idx]
                if target_type == 0:  # Numerical feature
                    # Regression task, directly use prediction value
                    pred_probs = pred
                else:  # Categorical feature
                    # Classification task
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        # Multi-class case, take probability of positive class
                        pred_probs = torch.softmax(pred, dim=-1)[:, 1]  # Take probability of class 1
                    else:
                        # Binary case
                        pred_probs = torch.sigmoid(pred) if pred.size(-1) == 1 else pred
            else:
                # Default to classification (backward compatibility)
                if pred.dim() > 1 and pred.size(-1) > 1:
                    pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                else:
                    pred_probs = torch.sigmoid(pred) if pred.size(-1) == 1 else pred
            
            # Store results
            if v is not None and target_idx < len(v) and v[target_idx] == 0:
                # Numerical feature
                all_pred_probs.extend(pred_probs.detach().cpu().numpy())
            else:
                # Categorical feature
                all_pred_probs.extend(pred_probs.detach().cpu().numpy())
            
            all_targets.extend(target[:, target_idx].cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_pred_probs = np.array(all_pred_probs)
    
    # Calculate AUC (binary case)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_targets, all_pred_probs)
    except Exception:
        auc = 0.0
    
    return auc

def compute_loss(model, x, y, v, num_classes_dict, alpha=1.0, beta=1.0, regression_weight=1.0, device='cuda', 
                prediction_target=None, prediction_idx=None, mask_positions=None, task_type='reconstruction', mask=None, binary_mask_path=None, mask_threshold=0.5):
    """
    Compute total loss, including reconstruction loss, sparsity loss, DAG constraint loss, and prediction loss (optional)
    
    Args:
        model: model
        x: input data
        y: reconstruction target
        v: feature type vector
        num_classes_dict: dictionary of categorical feature classes
        alpha: reconstruction loss weight
        beta: sparsity and DAG loss weight
        regression_weight: regression task loss weight, used to balance the magnitude difference between regression and classification tasks
        device: device
        prediction_target: prediction target label [B, 1] or None
        prediction_idx: prediction target feature index, None means no prediction
        mask_positions: masked positions [B, num_masked]
        task_type: task type, 'reconstruction' or 'prediction'
    """
    # Forward pass
    if prediction_target is not None and prediction_idx is not None:
        outputs, prediction, mask = model(x, mask_positions=mask_positions, target_idx=prediction_idx, task_type=task_type)
    else:
        outputs, prediction, mask = model(x, mask_positions=mask_positions, task_type=task_type)
        prediction = None
    
    # Calculate reconstruction loss
    reconstruction_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    if task_type == 'reconstruction' and mask_positions is not None:
        # Mask only one column per batch, all samples use the same mask position
        mask_pos = mask_positions[0, 0].item()  # Get mask position
        
        if mask_pos < len(outputs) and outputs[mask_pos] is not None:
            if v[mask_pos] == 0:  # Continuous feature
                target = y[:, mask_pos:mask_pos+1] if outputs[mask_pos].dim() > 1 else y[:, mask_pos]
                # Ensure target and outputs[mask_pos] have matching dimensions
                if outputs[mask_pos].dim() > 1:
                    loss = F.mse_loss(outputs[mask_pos], target)
                else:
                    loss = F.mse_loss(outputs[mask_pos].squeeze(), target)
                # Add weight for regression task
                loss = loss * regression_weight
                reconstruction_loss = loss
            else:  # Categorical feature
                # Get number of classes for this feature
                num_classes = num_classes_dict.get(mask_pos, 2)  # Default to 2 classes
                # Calculate CE loss and divide by log(C) to balance gradient magnitude for different number of classes
                loss = F.cross_entropy(outputs[mask_pos], y[:, mask_pos].long()) / torch.log(torch.tensor(num_classes, dtype=torch.float32, device=x.device))
                reconstruction_loss = loss
    elif task_type == 'reconstruction':
        # Calculate loss for all positions (backward compatibility)
        for i, output in enumerate(outputs):
            if v[i] == 0:  # Continuous feature
                target = y[:, i:i+1] if output.dim() > 1 else y[:, i]
                loss = F.mse_loss(output, target)
                # Add weight for regression task
                loss = loss * regression_weight
                reconstruction_loss = reconstruction_loss + loss
            else:  # Categorical feature
                # Get number of classes for this feature
                num_classes = num_classes_dict.get(i, 2)  # Default to 2 classes
                # Calculate CE loss and divide by log(C) to balance gradient magnitude for different number of classes
                loss = F.cross_entropy(output, y[:, i].long()) / torch.log(torch.tensor(num_classes, dtype=torch.float32, device=x.device))
                reconstruction_loss = reconstruction_loss + loss
    
    # Calculate sparsity loss (using group lasso, grouped by out-degree)
    sparsity_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    if task_type == 'reconstruction':
            # Do group lasso for each out-edge (column)
            # Each column represents the out-degree of a node, as a group
        col_groups = torch.sum(torch.abs(mask[0]), dim=0)  # L1 norm of each column [num_features]
            # Group lasso: sum of L2 norm for each column
        sparsity_loss = torch.sum(torch.sqrt(col_groups + 1e-8))  # Add small constant to avoid numerical issues
    
    # Calculate DAG constraint loss (using NOTEARS method)
    dag_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    if task_type == 'reconstruction':
        dag_loss = notears_dag_loss(mask[0])
    
    # Calculate regularization loss (alpha * reconstruction + beta * (sparsity + DAG))
    regularization_loss = alpha * reconstruction_loss + beta * (sparsity_loss + dag_loss)
    
    # Calculate prediction loss (if prediction target provided)
    prediction_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    if prediction_target is not None and prediction is not None:
        # Determine loss function based on feature type, not data type
        if v is not None and prediction_idx is not None and prediction_idx < len(v):
            target_type = v[prediction_idx]
            if target_type == 0:  # Numerical feature - regression
                prediction_loss = F.mse_loss(prediction.squeeze(-1), prediction_target.squeeze(-1))
                # Add weight for regression prediction task
                prediction_loss = prediction_loss * regression_weight
            else:  # Categorical feature - classification
                # For binary case, choose loss function based on output dimension
                if prediction.size(-1) == 1:
                    # Single value output, use BCEWithLogitsLoss (automatically applies sigmoid)
                    prediction_loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), prediction_target.squeeze(-1).float())
                else:
                    # Multi-value output [B, num_classes], use CrossEntropy (automatically applies softmax)
                    # Get number of classes for prediction target
                    num_classes = prediction.size(-1)
                    # Calculate CE loss and divide by log(C) to balance gradient magnitude for different number of classes
                    prediction_loss = F.cross_entropy(prediction, prediction_target.squeeze(-1).long()) / torch.log(torch.tensor(num_classes, dtype=torch.float32, device=x.device))
        else:
            # Default to classification
            if prediction.size(-1) == 1:
                prediction_loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), prediction_target.squeeze(-1).float())
            else:
                # Get number of classes for prediction target
                num_classes = prediction.size(-1)
                # Calculate CE loss and divide by log(C) to balance gradient magnitude for different number of classes
                prediction_loss = F.cross_entropy(prediction, prediction_target.squeeze(-1).long()) / torch.log(torch.tensor(num_classes, dtype=torch.float32, device=x.device))
    
    # Manually compute L2 regularization for CausalPredictor's feature_weights parameter
    feature_weights_l2_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    if task_type == 'prediction' and hasattr(model, 'predictor') and hasattr(model.predictor, 'feature_weights'):
        lambda_l2 = 1e-4  # Base regularization strength
        
        if mask is not None and prediction_idx is not None:
            # Calculate distance to target
            max_distance = 5
            distances = get_nodes_at_distance(mask[0], prediction_idx, max_distance=max_distance, threshold=mask_threshold, binary_mask_path=binary_mask_path)
            
            # Create distance weight vector [num_features]
            num_features = model.predictor.feature_weights.shape[0]
            distance_weights = torch.ones(num_features, device=x.device)
            
            # Record nodes assigned distance
            assigned_nodes = set()
            
            # Set weights based on distance: closer nodes have smaller regularization strength
            for dist, nodes in distances.items():
                if dist == 0:  # Target itself, distance 0, weakest regularization
                    for node in nodes:
                        if node < num_features:
                            distance_weights[node] = 0.0  # Distance 0, regularization weight 0
                            assigned_nodes.add(node)
                else:  # Other distances
                    for node in nodes:
                        if node < num_features:
                            distance_weights[node] = 1.0 * dist
                            assigned_nodes.add(node)
            
            # Points not calculated distance are considered n+1 (here n=3)
            for node in range(num_features):
                if node not in assigned_nodes:
                    distance_weights[node] = 1.0 * (max_distance + 1)  # Distance 4
            
            # Manually compute L2 regularization with distance weights
            weighted_l2_loss = lambda_l2 * torch.sum((model.predictor.feature_weights ** 2) * distance_weights)
            feature_weights_l2_loss = weighted_l2_loss
        else:
            # If no mask or prediction_idx, use original L2 regularization
            feature_weights_l2_loss = lambda_l2 * torch.sum(model.predictor.feature_weights ** 2)
        
        # Add L2 regularization to prediction_loss
        prediction_loss = prediction_loss + feature_weights_l2_loss
    
    # Calculate total loss
    total_loss = regularization_loss + prediction_loss
    
    loss_components = {
        'reconstruction_loss': reconstruction_loss.item(),
        'sparsity_loss': sparsity_loss.item(),
        'dag_loss': dag_loss.item(),
        'regularization_loss': regularization_loss.item(),
        'prediction_loss': prediction_loss.item() if prediction_target is not None else 0.0,
        'total_loss': total_loss.item()
    }
    
    return total_loss, regularization_loss, prediction_loss, loss_components

def should_update_batch(batch_idx, update_strategy):
    """
    Determine if current batch should be updated
    
    Args:
        batch_idx: current batch index
        update_strategy: (num, interval, offset) update strategy
        - num: number of batches to update
        - interval: interval size
        - offset: offset, 0 means update from the beginning, 1 means update from the end
        
    Returns:
        bool: whether to update
    """
    num, interval, offset = update_strategy
    
    if interval <= 0:
        return True  # Always update
    
    # Calculate current batch position in interval (starting from 0)
    position_in_interval = (batch_idx % interval)
    
    if offset == 0:
        # Update first num batches
        return position_in_interval < num
    else:
        # Update last num batches (from interval-num)
        return position_in_interval >= (interval - num)

def generate_random_mask_positions(batch_size, num_features, device='cuda'):
    """
    Generate random mask positions (mask only one column per batch)
    
    Args:
        batch_size: batch size
        num_features: number of features
        device: device
        
    Returns:
        torch.Tensor: shape [batch_size, 1] mask positions
    """
    # Generate a random position for the entire batch, all samples use the same mask position
    mask_position = torch.randint(0, num_features, (1,), device=device)
    mask_positions = mask_position.repeat(batch_size, 1)
    return mask_positions

def train_msk_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    v: torch.Tensor,
    num_classes_dict: Dict[int, int],
    test_loader: DataLoader = None,  # Add test set parameter
    # Hyperparameters
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    num_epochs: int = 30,
    patience: int = 7,
    # Loss weight parameters
    alpha: float = 1.0,      # Reconstruction loss weight
    beta: float = 1.0,       # Sparsity and DAG loss weight
    regression_weight: float = 1.0,  # Regression task loss weight
    # Update strategy parameters
    recon_update_strategy: tuple = (4, 5, 0),  # (num, interval, offset) reconstruction link update strategy
    pred_update_strategy: tuple = (4, 5, 0),   # (num, interval, offset) prediction link update strategy
    # Prediction related parameters
    prediction_idx=None,     # Prediction target feature index
    # Optimizer parameters
    optimizer_name: str = 'adam',
    scheduler_name: str = 'reduce_on_plateau',
    # Other parameters
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Optional[str] = None,
    log_interval: int = 10,
    binary_mask_path: Optional[str] = None,  # File path to store binary_adj
    log_infer_time: bool = True,  # Whether to output inference time
    mask_threshold: float = 0.5,  # Threshold to binarize causal mask (hierarchy/distance)
) -> nn.Module:
    """
    Main function to train the model, supporting reconstruction and prediction tasks (MSK version)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Create training log directory
        log_dir = save_dir / 'training_logs'
        log_dir.mkdir(exist_ok=True)
    
    # Move model to specified device
    model = model.to(device)
    
    # Select optimizer
    if optimizer_name.lower() == 'adam':
        # Separate different types of parameters
        mask_params = [p for n, p in model.named_parameters() if 'mask_logits' in n]
        attention_params = [p for n, p in model.named_parameters() if any(x in n for x in ['q_proj', 'k_proj', 'v_proj'])]
        feature_weights_params = [p for n, p in model.named_parameters() if 'feature_weights' in n]
        predictor_params = [p for n, p in model.named_parameters() 
                          if 'predictor' in n and 'feature_weights' not in n]
        other_params = [p for n, p in model.named_parameters() 
                       if 'mask_logits' not in n and not any(x in n for x in ['q_proj', 'k_proj', 'v_proj']) 
                       and 'predictor' not in n]
        
        optimizer = optim.Adam([
            {'params': other_params, 'weight_decay': weight_decay},
            {'params': mask_params, 'weight_decay': 0.0},
            {'params': attention_params, 'weight_decay': 0.0},
            {'params': predictor_params, 'weight_decay': weight_decay},
            {'params': feature_weights_params, 'weight_decay': 0.0}  # feature_weights do not use weight_decay
        ], lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        # Separate different types of parameters
        mask_params = [p for n, p in model.named_parameters() if 'mask_logits' in n]
        attention_params = [p for n, p in model.named_parameters() if any(x in n for x in ['q_proj', 'k_proj', 'v_proj'])]
        feature_weights_params = [p for n, p in model.named_parameters() if 'feature_weights' in n]
        predictor_params = [p for n, p in model.named_parameters() 
                          if 'predictor' in n and 'feature_weights' not in n]
        other_params = [p for n, p in model.named_parameters() 
                       if 'mask_logits' not in n and not any(x in n for x in ['q_proj', 'k_proj', 'v_proj']) 
                       and 'predictor' not in n]
        
        optimizer = optim.SGD([
            {'params': other_params, 'weight_decay': weight_decay},
            {'params': mask_params, 'weight_decay': 0.0},
            {'params': attention_params, 'weight_decay': 0.0},
            {'params': predictor_params, 'weight_decay': weight_decay},
            {'params': feature_weights_params, 'weight_decay': 0.0}  # feature_weights do not use weight_decay
        ], lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Select learning rate scheduler
    if scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mask_generator=model.mask_generator, use_prediction_loss=True)
    
    # Pure training time per epoch (excludes validation/test/metric passes), for runtime scaling experiments
    epoch_train_times = []

    # Training loop
    for epoch in range(num_epochs):
        # Update parameters
        model.update_parameters()
        current_params = model.get_parameters()
        
        # Training phase
        model.train()
        
        # For storing loss from the last update
        last_total_loss = 0
        last_reg_loss = 0
        last_pred_loss = 0
        
        # Record training start time
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Get prediction target from target (if prediction index specified)
            batch_prediction_target = None
            if prediction_idx is not None:
                batch_prediction_target = target[:, prediction_idx]
            
            # Training phase
            # 1. First perform reconstruction task
            mask_positions = generate_random_mask_positions(data.size(0), data.size(1), device=device)
            
            # Check if reconstruction link should be updated (stop updating after 10 epochs)
            should_update_recon = should_update_batch(batch_idx, recon_update_strategy) and epoch < 15
            
            if should_update_recon:
                optimizer.zero_grad()
                
                # Freeze feature_weights parameters in reconstruction link
                if hasattr(model, 'predictor') and hasattr(model.predictor, 'feature_weights'):
                    model.predictor.feature_weights.requires_grad_(False)
                
                outputs, prediction, mask = model(data, mask_positions=mask_positions, task_type='reconstruction')
                total_loss, reg_loss, pred_loss, loss_components = compute_loss(
                    model, data, target, v, num_classes_dict, 
                        alpha, beta, regression_weight, device,
                        batch_prediction_target, prediction_idx,
                    mask_positions=mask_positions, task_type='reconstruction', mask=mask,
                    mask_threshold=mask_threshold
                )
                # Backpropagate using total loss
                total_loss.backward()
                optimizer.step()
                
                # Restore gradient calculation for feature_weights parameters
                if hasattr(model, 'predictor') and hasattr(model.predictor, 'feature_weights'):
                    model.predictor.feature_weights.requires_grad_(True)
            
                # Record loss from last update
                last_total_loss = total_loss.item()
                last_reg_loss = reg_loss.item()
                last_pred_loss = pred_loss.item()
            
            # 2. Then perform prediction task (if prediction target provided)
            if prediction_idx is not None:
                # Check if prediction link should be updated
                should_update_pred = should_update_batch(batch_idx, pred_update_strategy)
                
                if should_update_pred:
                    optimizer.zero_grad()
                    
                    # Freeze mask-related parameters in prediction link
                    mask_params_frozen = []
                    for name, param in model.named_parameters():
                        if 'mask_logits' in name or 'mask_generator' in name:
                            if param.requires_grad:
                                param.requires_grad_(False)
                                mask_params_frozen.append(name)
                    
                    outputs_pred, prediction_pred, mask_pred = model(data, target_idx=prediction_idx, task_type='prediction')
                    total_loss_pred, reg_loss_pred, pred_loss_pred, loss_components_pred = compute_loss(
                        model, data, target, v, num_classes_dict, 
                            alpha, beta, regression_weight, device,
                            batch_prediction_target, prediction_idx,
                        task_type='prediction', mask=mask_pred, binary_mask_path=binary_mask_path,
                        mask_threshold=mask_threshold
                    )
                    # Backpropagate using only prediction loss
                    pred_loss_pred.backward()
                    optimizer.step()
                    
                    # Restore gradient calculation for mask parameters
                    for name in mask_params_frozen:
                        for param_name, param in model.named_parameters():
                            if param_name == name:
                                param.requires_grad_(True)
                                break
            
            # Record data for each batch
            if batch_idx % log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                          f'Total Loss: {last_total_loss:.6f}\t'
                          f'Reg Loss: {last_reg_loss:.6f}\t'
                          f'Pred Loss: {last_pred_loss:.6f}')
        
        # Calculate and record training time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_train_times.append(epoch_duration)
        logger.info(f'Epoch {epoch} time consumption: {epoch_duration:.2f}s')
            
        # Validation phase
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_sparse_loss = 0
        val_dag_loss = 0
        val_reg_loss = 0
        val_pred_loss = 0
        
        # Variables for calculating accuracy and AUC (if prediction task)
        val_correct = 0
        val_total = 0
        all_val_targets = []
        all_val_pred_probs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                
                # Get prediction target from target (if prediction index specified)
                batch_prediction_target = None
                if prediction_idx is not None:
                    batch_prediction_target = target[:, prediction_idx]
                
                # During validation, only evaluate prediction task
                outputs, prediction, mask = model(data, target_idx=prediction_idx, task_type='prediction')
                total_loss, reg_loss, pred_loss, loss_components = compute_loss(
                    model, data, target, v, num_classes_dict, 
                    alpha, beta, regression_weight, device,
                    batch_prediction_target, prediction_idx,
                    task_type='prediction', mask=mask,
                    mask_threshold=mask_threshold
                )
                val_loss += total_loss.item()
                val_recon_loss += loss_components['reconstruction_loss']
                val_sparse_loss += loss_components['sparsity_loss']
                val_dag_loss += loss_components['dag_loss']
                val_reg_loss += loss_components['regularization_loss']
                val_pred_loss += loss_components['prediction_loss']
                
                # Calculate accuracy and AUC simultaneously (if prediction task)
                if prediction_idx is not None:
                    pred = prediction.squeeze(-1)
                    target_values = target[:, prediction_idx]
                    
                    # Process prediction results based on target feature type (consistent with validation set logic)
                    if v is not None and prediction_idx < len(v):
                        target_type = v[prediction_idx]
                        if target_type == 0:  # Numerical feature
                            pred_probs = pred
                            pred_classes = pred
                        else:  # Categorical feature
                            if pred.dim() > 1 and pred.size(-1) > 1:
                                # Multi-value output [B, 2], use softmax to get probability, argmax to get class
                                pred_probs = torch.softmax(pred, dim=-1)[:, 1]  # Take probability of class 1
                                pred_classes = pred.argmax(dim=-1)
                            else:
                                # Single value output, use sigmoid
                                pred_probs = torch.sigmoid(pred)
                                pred_classes = (pred_probs > 0.5).long().view(-1)
                    else:
                        # Default to classification
                        if pred.dim() > 1 and pred.size(-1) > 1:
                            pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                            pred_classes = pred.argmax(dim=-1)
                        else:
                            pred_probs = torch.sigmoid(pred)
                            pred_classes = (pred_probs > 0.5).long().view(-1)
                    
                    # Calculate accuracy
                    val_correct += (pred_classes == target_values).sum().item()
                    val_total += target_values.size(0)
                    
                    # Store data for AUC calculation
                    all_val_pred_probs.extend(pred_probs.detach().cpu().numpy())
                    all_val_targets.extend(target_values.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_sparse_loss /= len(val_loader)
        val_dag_loss /= len(val_loader)
        val_reg_loss /= len(val_loader)
        val_pred_loss /= len(val_loader)
        
        # Calculate validation accuracy and AUC (if prediction task)
        val_accuracy = 0.0
        val_auc = 0.0
        val_mse = 0.0  # Add MSE for regression task
        if prediction_idx is not None:
            # Determine task type
            is_regression = v is not None and prediction_idx < len(v) and v[prediction_idx] == 0
            
            if is_regression:
                # Regression task: calculate MSE
                from sklearn.metrics import mean_squared_error
                val_mse = mean_squared_error(all_val_targets, all_val_pred_probs)
                val_accuracy = 0.0  # Regression task has no accuracy concept
                val_auc = 0.0       # Regression task has no AUC concept
            else:
                # Classification task: calculate accuracy and AUC
                val_accuracy = val_correct / val_total if val_total > 0 else 0.0
                
                # Calculate validation AUC
                all_val_targets = np.array(all_val_targets)
                all_val_pred_probs = np.array(all_val_pred_probs)
            
            try:
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(all_val_targets, all_val_pred_probs)
            except Exception as e:
                val_auc = 0.0
        
        # Calculate training accuracy (if prediction task)
        train_accuracy = 0.0
        train_mse = 0.0  # Add training MSE
        if prediction_idx is not None:
            train_correct = 0
            train_total = 0
            all_train_targets = []
            all_train_pred_probs = []
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                outputs, prediction, mask = model(data, target_idx=prediction_idx, task_type='prediction')
                pred = prediction.squeeze(-1)
                target_values = target[:, prediction_idx]
                
                # Process prediction results based on target feature type (consistent with validation set logic)
                if v is not None and prediction_idx < len(v):
                    target_type = v[prediction_idx]
                    if target_type == 0:  # Numerical feature
                        pred_classes = pred
                        pred_probs = pred
                    else:  # Categorical feature
                        if pred.dim() > 1 and pred.size(-1) > 1:
                            # Multi-value output [B, 2], use argmax to get class
                            pred_classes = pred.argmax(dim=-1)
                            pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                        else:
                            # Single value output, use sigmoid
                            pred_probs = torch.sigmoid(pred)
                            pred_classes = (pred_probs > 0.5).long().view(-1)
                else:
                    # Default to classification
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        pred_classes = pred.argmax(dim=-1)
                        pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                    else:
                        pred_probs = torch.sigmoid(pred)
                        pred_classes = (pred_probs > 0.5).long().view(-1)
                
                train_correct += (pred_classes == target_values).sum().item()
                train_total += target_values.size(0)
                
                # Store data for MSE calculation
                all_train_targets.extend(target_values.cpu().numpy())
                all_train_pred_probs.extend(pred_probs.detach().cpu().numpy())
            
            # Determine task type and calculate corresponding metrics
            is_regression = v is not None and prediction_idx < len(v) and v[prediction_idx] == 0
            if is_regression:
                # Regression task: calculate MSE
                from sklearn.metrics import mean_squared_error
                train_mse = mean_squared_error(all_train_targets, all_train_pred_probs)
                train_accuracy = 0.0  # Regression task has no accuracy concept
            else:
                # Classification task: calculate accuracy
                train_accuracy = train_correct / train_total if train_total > 0 else 0.0
                train_mse = 0.0  # Classification task has no MSE concept
        
        # Update learning rate
        if scheduler_name.lower() == 'reduce_on_plateau':
            # Learning rate scheduling based on prediction loss
            scheduler.step(val_pred_loss)
        else:
            scheduler.step()
        
        # Test set evaluation (if test set provided)
        test_results = None
        if test_loader is not None and prediction_idx is not None:
            test_results = evaluate_test_set(model, test_loader, prediction_idx, device, v, log_infer_time)
        
        # Print training information
        if prediction_idx is not None:
            # Determine task type
            is_regression = v is not None and prediction_idx < len(v) and v[prediction_idx] == 0
            
            if is_regression:
                # Regression task output
                test_info = f'Test MSE: {test_results["mse"]:.6f}' if test_results else ''
                logger.info(f'Epoch: {epoch}\t'
                           f'Val Loss: {val_loss:.6f}\t'
                           f'Train MSE: {train_mse:.6f}\t'
                           f'Val MSE: {val_mse:.6f}\t'
                           f'{test_info}')
            else:
                # Classification task output
                test_info = f'Test Acc: {test_results["acc"]:.4f} Test AUC: {test_results["auc"]:.4f}' if test_results else ''
                logger.info(f'Epoch: {epoch}\t'
                            f'Val Loss: {val_loss:.6f}\t'
                            f'Train Acc: {train_accuracy:.4f}\t'
                            f'Val Acc: {val_accuracy:.4f}\t'
                            f'Val AUC: {val_auc:.4f}\t'
                            f'{test_info}')
        else:
            logger.info(f'Epoch: {epoch}\t'
                       f'Val Loss: {val_loss:.6f}\t'
                       f'Last Total Loss: {last_total_loss:.6f}\t'
                       f'Last Reg Loss: {last_reg_loss:.6f}\t'
                       f'Last Pred Loss: {last_pred_loss:.6f}')
        
        # Save best model
        if save_dir:
            # Save best model based on prediction loss
            if val_pred_loss < early_stopping.val_loss_min:
                early_stopping.val_loss_min = val_pred_loss
                save_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists (in case it was cleaned mid-run)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_prediction_loss': val_pred_loss,
                    'mask_params': current_params,
                }, save_dir / 'best_model.pth')
        
        # Early stopping check
        early_stopping(val_loss, val_pred_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    model.epoch_train_times = epoch_train_times

    # Load best model (fall back to the final state if the checkpoint was cleaned mid-run)
    if save_dir:
        best_model_path = save_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logging.warning(f"best_model.pth not found at {best_model_path}; using the final model state instead.")

    return model 

def evaluate_test_set(model, test_loader, target_idx, device, v=None, log_infer_time=False):
    """
    Evaluate test set performance, return AUC, ACC (classification) or MSE (regression)
    
    Args:
        model: model
        test_loader: test data loader
        target_idx: target feature index
        device: device
        v: feature type vector, used to determine target feature type
        log_infer_time: whether to output inference time
    
    Returns:
        dict: dictionary containing test metrics
    """
    import time
    import logging
    logger = logging.getLogger(__name__)
    model.eval()
    all_targets = []
    all_pred_probs = []
    all_pred_classes = []
    
    # Statistic inference time
    start_time = time.time()
    num_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_samples += data.size(0)
            # Get prediction results
            outputs, prediction, mask = model(data, target_idx=target_idx, task_type='prediction')
            pred = prediction.squeeze(-1)
            target_values = target[:, target_idx]
            
            # Process prediction results based on target feature type
            if v is not None and target_idx < len(v):
                target_type = v[target_idx]
                if target_type == 0:  # Numerical feature - regression
                    pred_probs = pred
                    pred_classes = pred  # Regression task has no class concept
                else:  # Categorical feature
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        # Multi-class case
                        pred_probs = torch.softmax(pred, dim=-1)[:, 1]  # Take probability of class 1
                        pred_classes = pred.argmax(dim=-1)
                    else:
                        # Binary case
                        pred_probs = torch.sigmoid(pred) if pred.size(-1) == 1 else pred
                        pred_classes = (pred_probs > 0.5).long().view(-1)
            else:
                # Default to classification
                if pred.dim() > 1 and pred.size(-1) > 1:
                    pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                    pred_classes = pred.argmax(dim=-1)
                else:
                    pred_probs = torch.sigmoid(pred) if pred.size(-1) == 1 else pred
                    pred_classes = (pred_probs > 0.5).long().view(-1)
            
            # Store results
            all_pred_probs.extend(pred_probs.detach().cpu().numpy())
            all_pred_classes.extend(pred_classes.detach().cpu().numpy())
            all_targets.extend(target_values.cpu().numpy())
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_samples if num_samples > 0 else 0.0
    if log_infer_time:
        logger.info(f"[Test] Total inference time: {total_time:.4f}s, average time per sample: {avg_time*1000:.4f}ms")
    
    all_targets = np.array(all_targets)
    all_pred_probs = np.array(all_pred_probs)
    all_pred_classes = np.array(all_pred_classes)
    
    # Determine task type
    is_regression = v is not None and target_idx < len(v) and v[target_idx] == 0
    
    results = {}
    
    if is_regression:
        # Regression task: calculate MSE
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(all_targets, all_pred_probs)
        results['mse'] = mse
        results['auc'] = 0.0  # Regression task has no AUC
        results['acc'] = 0.0  # Regression task has no accuracy
    else:
        # Classification task: calculate AUC and accuracy
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            auc = roc_auc_score(all_targets, all_pred_probs)
            acc = accuracy_score(all_targets, all_pred_classes)
        except Exception:
            auc = 0.0
            acc = 0.0
        
        results['auc'] = auc
        results['acc'] = acc
        results['mse'] = 0.0  # Classification task has no MSE
    
    return results 