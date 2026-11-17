import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def evaluate_prediction_accuracy(model, data_loader, target_idx, device, v=None):
    """
    Evaluate prediction accuracy (supports classification and regression)
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_pred_probs = []
    infer_start = time.time()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = model(data, target_idx=target_idx)
            pred = prediction.squeeze(-1)
            target_values = target.squeeze(-1)
            
            if v is not None and target_idx < len(v):
                target_type = v[target_idx]
                if target_type == 0:
                    pred_values = pred
                    pred_labels = pred
                else:
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                        pred_labels = pred.argmax(dim=-1)
                    else:
                        pred_probs = torch.sigmoid(pred)
                        pred_labels = (pred_probs > 0.5).long().view(-1)
            else:
                if pred.dim() > 1 and pred.size(-1) > 1:
                    pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                    pred_labels = pred.argmax(dim=-1)
                else:
                    pred_probs = torch.sigmoid(pred)
                    pred_labels = (pred_probs > 0.5).long().view(-1)
            
            if v is not None and target_idx < len(v) and v[target_idx] == 0:
                all_pred_probs.extend(pred_values.detach().cpu().numpy())
                all_predictions.extend(pred_values.detach().cpu().numpy())
            else:
                all_pred_probs.extend(pred_probs.detach().cpu().numpy())
                all_predictions.extend(pred_labels.detach().cpu().numpy())
            all_targets.extend(target_values.cpu().numpy())
    infer_end = time.time()
    infer_time = infer_end - infer_start
    num_samples = len(data_loader.dataset)
    logging.info(f"Inference total time: {infer_time:.2f} seconds, average per sample: {infer_time/num_samples:.4f} seconds")
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_pred_probs = np.array(all_pred_probs)
    
    if v is not None and target_idx < len(v) and v[target_idx] == 0:
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        }
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary'
        )
        accuracy = accuracy_score(all_targets, all_predictions)
        
        try:
            auc = roc_auc_score(all_targets, all_pred_probs)
        except Exception:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    v: torch.Tensor,
    num_classes_dict: Dict[int, int],
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    num_epochs: int = 30,
    patience: int = 7,
    prediction_idx=None,
    optimizer_name: str = 'adam',
    scheduler_name: str = 'reduce_on_plateau',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Optional[str] = None,
    log_interval: int = 10
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Main function to train the baseline model
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    if scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif scheduler_name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")
    
    early_stopping = EarlyStopping(patience=patience)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_mse': [],
        'val_mse': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            batch_prediction_target = target.squeeze(-1)
            
            optimizer.zero_grad()
            
            prediction = model(data, target_idx=prediction_idx)
            
            if prediction_idx is not None and batch_prediction_target is not None:
                if v is not None and prediction_idx < len(v):
                    target_type = v[prediction_idx]
                    if target_type == 0:
                        loss = F.mse_loss(prediction.squeeze(-1), batch_prediction_target)
                    else:
                        if prediction.size(-1) == 1:
                            loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), batch_prediction_target.float())
                        else:
                            loss = F.cross_entropy(prediction, batch_prediction_target.long())
                else:
                    if prediction.size(-1) == 1:
                        loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), batch_prediction_target.float())
                    else:
                        loss = F.cross_entropy(prediction, batch_prediction_target.long())
            else:
                loss = F.mse_loss(prediction.squeeze(-1), target.squeeze(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f}')
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f'Epoch {epoch} time consumption: {epoch_duration:.2f}s')
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                batch_prediction_target = target.squeeze(-1)
                
                prediction = model(data, target_idx=prediction_idx)
                
                if prediction_idx is not None and batch_prediction_target is not None:
                    if v is not None and prediction_idx < len(v):
                        target_type = v[prediction_idx]
                        if target_type == 0:
                            loss = F.mse_loss(prediction.squeeze(-1), batch_prediction_target)
                        else:
                            if prediction.size(-1) == 1:
                                loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), batch_prediction_target.float())
                            else:
                                loss = F.cross_entropy(prediction, batch_prediction_target.long())
                    else:
                        if prediction.size(-1) == 1:
                            loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), batch_prediction_target.float())
                        else:
                            loss = F.cross_entropy(prediction, batch_prediction_target.long())
                else:
                    loss = F.mse_loss(prediction.squeeze(-1), target.squeeze(-1))
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_metrics = evaluate_prediction_accuracy(model, train_loader, prediction_idx, device, v)
        val_metrics = evaluate_prediction_accuracy(model, val_loader, prediction_idx, device, v)
        
        if scheduler_name.lower() == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        is_regression = v is not None and prediction_idx is not None and prediction_idx < len(v) and v[prediction_idx] == 0
        if is_regression:
            history['train_mse'].append(train_metrics['mse'])
            history['val_mse'].append(val_metrics['mse'])
            history['train_accuracy'].append(0.0)
            history['val_accuracy'].append(0.0)
        else:
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['train_mse'].append(0.0)
            history['val_mse'].append(0.0)
        
        if is_regression:
            logger.info(f'Epoch: {epoch}\t'
                       f'Train Loss: {train_loss:.6f}\t'
                       f'Val Loss: {val_loss:.6f}\t'
                       f'Train MSE: {train_metrics["mse"]:.6f}\t'
                       f'Val MSE: {val_metrics["mse"]:.6f}\t'
                       f'Val R²: {val_metrics["r2"]:.4f}')
        else:
            logger.info(f'Epoch: {epoch}\t'
                       f'Train Loss: {train_loss:.6f}\t'
                       f'Val Loss: {val_loss:.6f}\t'
                       f'Train Acc: {train_metrics["accuracy"]:.4f}\t'
                       f'Val Acc: {val_metrics["accuracy"]:.4f}\t'
                       f'Val AUC: {val_metrics["auc"]:.4f}')
        

        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    return model, history 