import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.causal_attention_msk_model import CausalAttentionMskModel
from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset

from datasets.adult_dataset import load_adult_reconstruct_dataset
from datasets.cardio_dataset import load_cardio_reconstruct_dataset
from datasets.diamonds_dataset import load_diamonds_reconstruct_dataset
from datasets.elevator_dataset import load_elevator_reconstruct_dataset
from datasets.housesale_dataset import load_housesale_reconstruct_dataset
from datasets.crime_dataset import load_crime_reconstruct_dataset
from datasets.meps_dataset import load_meps_reconstruct_dataset
from torch.utils.data import DataLoader, random_split
# Default import train_msk_model, but can be dynamically selected via parameters
from utils.train_msk_utils import train_msk_model
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from models.mask_generators import MaskGenerator, GumbelSoftmaxMaskGenerator, SigmoidMaskGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import argparse

print(f"PyTorch version: {torch.__version__}")

# Use different versions of models
def get_model_class(model_source="models.causal_attention_msk_model"):
    """
    Dynamically import model class based on model_source
    
    Args:
        model_source: model module path, default "models.causal_attention_msk_model"
    
    Returns:
        model class
    """
    try:
        # Dynamically import module, class name fixed as CausalAttentionMskModel
        module = __import__(model_source, fromlist=['CausalAttentionMskModel'])
        return getattr(module, 'CausalAttentionMskModel')
    except ImportError as e:
        raise ImportError(f"Cannot import model module {model_source}: {e}")
    except AttributeError as e:
        raise AttributeError(f"CausalAttentionMskModel class not found in module {model_source}: {e}")

def get_train_function(train_source="utils.train_msk_utils"):
    """
    Dynamically import training function based on train_source
    
    Args:
        train_source: training module path, default "utils.train_msk_utils"
    
    Returns:
        training function
    """
    try:
        # Dynamically import module, function name fixed as train_msk_model
        module = __import__(train_source, fromlist=['train_msk_model'])
        return getattr(module, 'train_msk_model')
    except ImportError as e:
        raise ImportError(f"Cannot import training module {train_source}: {e}")
    except AttributeError as e:
        raise AttributeError(f"train_msk_model function not found in module {train_source}: {e}")


def evaluate_prediction_accuracy(model, data_loader, target_idx, device, v=None):
    """
    Evaluate prediction accuracy (supports classification and regression)
    
    Args:
        model: trained model
        data_loader: data loader
        target_idx: target feature index
        device: device
        v: feature type vector, used to determine target feature type
    
    Returns:
        dict: dictionary containing various evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_pred_probs = []  # Store prediction probabilities for AUC calculation
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # Use MSK embedding during testing, pass target_idx to ensure target column is masked
            outputs, prediction, mask = model(data, task_type='prediction', target_idx=target_idx)
            pred = prediction.squeeze(-1)
            
            # Process prediction results based on target feature type
            if v is not None and target_idx < len(v):
                target_type = v[target_idx]
                if target_type == 0:  # Numerical feature
                    # Regression task, directly use prediction value
                    pred_values = pred
                    pred_labels = pred  # Regression task has no label concept
                else:  # Categorical feature
                    # Categorical task
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        # Multi-class case, take probability of positive class
                        pred_probs = torch.softmax(pred, dim=-1)[:, 1]  # Take probability of class 1
                        pred_labels = pred.argmax(dim=-1)
                    else:
                        # Binary case, use sigmoid (corresponding to BCEWithLogitsLoss during training)
                        pred_probs = torch.sigmoid(pred)
                        pred_labels = (pred_probs > 0.5).long().view(-1)
            else:
                # Default to classification (backward compatibility)
                if pred.dim() > 1 and pred.size(-1) > 1:
                    pred_probs = torch.softmax(pred, dim=-1)[:, 1]
                    pred_labels = pred.argmax(dim=-1)
                else:
                    # Binary case, use sigmoid
                    pred_probs = torch.sigmoid(pred)
                    pred_labels = (pred_probs > 0.5).long().view(-1)
            
            # Store results
            if v is not None and target_idx < len(v) and v[target_idx] == 0:
                # Numerical feature
                all_pred_probs.extend(pred_values.detach().cpu().numpy())
                all_predictions.extend(pred_values.detach().cpu().numpy())
            else:
                # Categorical feature
                all_pred_probs.extend(pred_probs.detach().cpu().numpy())
                all_predictions.extend(pred_labels.detach().cpu().numpy())
            
            all_targets.extend(target[:, target_idx].cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_pred_probs = np.array(all_pred_probs)
    
    # Calculate different evaluation metrics based on target feature type
    if v is not None and target_idx < len(v) and v[target_idx] == 0:
        # Numerical feature - Regression task
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
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
        # Categorical feature - Classification task
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary'
        )
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Calculate AUC (binary case)
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

def test_causal_attention_msk_model(prefix="", dataset="housesale", num_heads=4, num_layers=1, gpu_id=-1,
                                   learning_rate=1e-4, weight_decay=1e-5, num_epochs=100, patience=100,
                                   alpha=1.0, beta=0.001, regression_weight=5.0,
                                   recon_update_strategy=(2, 10, 1), pred_update_strategy=(8, 10, 0),
                                   optimizer_name='adam', scheduler_name='reduce_on_plateau',
                                   d_model=64, dropout=0.1, batch_size=128, binary_mask_path=None, mask_threshold=0.5, 
                                   model_source="models.causal_attention_msk_model", train_source="utils.train_msk_utils",
                                   seed=42):
    """
    Test causal attention MSK model
    
    Args:
        prefix (str): prefix for result files, to distinguish different experiments
        dataset (str): dataset selection (e.g., "creditcard")
    """
    # Set random seed to ensure reproducible results
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cuda')
        # Set default GPU
        if gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
    
    # Select dataset based on parameter
    logger.info(f"Loading dataset: {dataset}")
    if dataset.lower() == "creditcard":
        dataset_obj, v, num_classes_dict = load_creditcard_reconstruct_dataset()
        dataset_name = "CreditCard"
    elif dataset.lower() == "adult":
        dataset_obj, v, num_classes_dict = load_adult_reconstruct_dataset()
        dataset_name = "Adult"
    elif dataset.lower() == "cardio":
        dataset_obj, v, num_classes_dict = load_cardio_reconstruct_dataset()
        dataset_name = "Cardio"
    elif dataset.lower() == "diamonds":
        dataset_obj, v, num_classes_dict = load_diamonds_reconstruct_dataset()
        dataset_name = "Diamonds"
    elif dataset.lower() == "elevator":
        dataset_obj, v, num_classes_dict = load_elevator_reconstruct_dataset()
        dataset_name = "Elevator"
    elif dataset.lower() == "housesale":
        dataset_obj, v, num_classes_dict = load_housesale_reconstruct_dataset()
        dataset_name = "Housesale"
    elif dataset.lower() == "crime":
        dataset_obj, v, num_classes_dict = load_crime_reconstruct_dataset()
        dataset_name = "Crime"
    elif dataset.lower() == "meps":
        dataset_obj, v, num_classes_dict = load_meps_reconstruct_dataset()
        dataset_name = "MEPS"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}, supported datasets: creditcard, adult, cardio, diamonds, elevator, housesale, crime, meps")
    
    # Ensure v is torch.Tensor type and move to correct device
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)
    
    # Print dataset information
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Dataset size: {len(dataset_obj)}")
    logger.info(f"Feature type vector v: {v}")
    logger.info(f"Shape of v: {v.shape}")
    logger.info(f"Number of categorical feature classes: {num_classes_dict}")
    
    # Get a sample to check dimensions
    sample_x, sample_y = dataset_obj[0]
    logger.info(f"Sample input shape: {sample_x.shape}")
    logger.info(f"Sample target shape: {sample_y.shape}")
    
    # Determine prediction target (last column)
    target_idx = len(v) - 1  # Last column is the prediction target
    logger.info(f"Prediction target index: {target_idx}")
    
    # Unified split: deterministic random shuffle -> 80/20 train-val/test -> 80/20 train/val
    from utils.split_utils import split_indices
    total_size = len(dataset_obj)
    train_idx, val_idx, test_idx = split_indices(total_size, seed=seed)
    train_dataset = torch.utils.data.Subset(dataset_obj, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(dataset_obj, val_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset_obj, test_idx.tolist())
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Original dataset size: {total_size}")
    logger.info(f"Training set size: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    logger.info(f"Validation set size: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    logger.info(f"Test set size: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    logger.info(f"Training + Validation set size: {len(train_dataset) + len(val_dataset)} ({(len(train_dataset) + len(val_dataset))/total_size*100:.1f}%)")
    
    # Initialize mask generator
    logger.info("\nInitializing mask generator...")
    mask_generator = SigmoidMaskGenerator(
        num_features=len(v),
        initial_threshold=0.2,
        final_threshold=0.2,
        threshold_multiplier=1.1
    ).to(device)
    
    # Dynamically import model class
    logger.info(f"\nImporting model class from {model_source}...")
    ModelClass = get_model_class(model_source)
    
    # Dynamically import training function
    logger.info(f"\nImporting training function from {train_source}...")
    train_function = get_train_function(train_source)
    
    # Initialize model
    logger.info("\nInitializing causal attention MSK model...")
    model = ModelClass(
        v=v,  # Feature type vector
        num_classes_dict=num_classes_dict,
        d_model=d_model,
        num_heads=num_heads,  # Use passed parameters
        num_layers=num_layers,  # Use passed parameters
        dropout=dropout,
        share_embedding=True,
        mask_generator=mask_generator,
        target_idx=target_idx
    ).to(device)
    
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Number of attention heads: {num_heads}")
    logger.info(f"Number of Transformer layers: {num_layers}")
    
    # Train model
    logger.info("\nStarting model training...")
    trained_model = train_function(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        v=v,
        num_classes_dict=num_classes_dict,
        test_loader=test_loader,  # Add test set parameter
        # Prediction related parameters
        prediction_idx=target_idx,
        # Training parameters
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        # Loss weight parameters
        alpha=alpha,
        beta=beta,
        regression_weight=regression_weight,
        # Update strategy parameters
        recon_update_strategy=recon_update_strategy,
        pred_update_strategy=pred_update_strategy,
        # Optimizer parameters
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        # Other parameters
        device=device,
        save_dir=f'checkpoints_causal_msk_{prefix}' if prefix else 'checkpoints_causal_msk',  # Save model to new directory
        log_interval=100,
        binary_mask_path=binary_mask_path,
        mask_threshold=mask_threshold
    )
    
    # Evaluate model performance
    logger.info("\nEvaluating model performance...")
    
    # Determine task type
    is_regression = v is not None and target_idx < len(v) and v[target_idx] == 0
    task_type = "Regression task" if is_regression else "Classification task"
    logger.info(f"Task type: {task_type}")
    
    # Training set evaluation
    train_metrics = evaluate_prediction_accuracy(trained_model, train_loader, target_idx, device, v)
    logger.info(f"Training set performance:")
    if is_regression:
        logger.info(f"  MSE: {train_metrics['mse']:.4f}")
        logger.info(f"  MAE: {train_metrics['mae']:.4f}")
        logger.info(f"  R²: {train_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {train_metrics['rmse']:.4f}")
    else:
        logger.info(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {train_metrics['precision']:.4f}")
        logger.info(f"  Recall: {train_metrics['recall']:.4f}")
        logger.info(f"  F1 score: {train_metrics['f1_score']:.4f}")
        logger.info(f"  AUC: {train_metrics['auc']:.4f}")
    
    # Validation set evaluation
    val_metrics = evaluate_prediction_accuracy(trained_model, val_loader, target_idx, device, v)
    logger.info(f"Validation set performance:")
    if is_regression:
        logger.info(f"  MSE: {val_metrics['mse']:.4f}")
        logger.info(f"  MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  R²: {val_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {val_metrics['rmse']:.4f}")
    else:
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall: {val_metrics['recall']:.4f}")
        logger.info(f"  F1 score: {val_metrics['f1_score']:.4f}")
        logger.info(f"  AUC: {val_metrics['auc']:.4f}")
    
    # Test set evaluation (final performance evaluation)
    test_metrics = evaluate_prediction_accuracy(trained_model, test_loader, target_idx, device, v)
    logger.info(f"Test set performance (final evaluation):")
    if is_regression:
        logger.info(f"  MSE: {test_metrics['mse']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")
        logger.info(f"  R²: {test_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    else:
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1 score: {test_metrics['f1_score']:.4f}")
        logger.info(f"  AUC: {test_metrics['auc']:.4f}")
    
    # Build file names (add prefix)
    mask_filename = f"{prefix}_causal_attention_msk_mask.png" if prefix else "causal_attention_msk_mask.png"
    results_filename = f"{prefix}_causal_attention_msk_results.txt" if prefix else "causal_attention_msk_results.txt"
    
    # Get final causal mask
    causal_mask = trained_model.get_causal_mask()
    
    logger.info(f"\nMask matrix sparsity: {(causal_mask > 0.5).mean():.2f}")
    logger.info(f"Mask matrix example:\n{causal_mask[:5, :5]}")
    
    # Save mask matrix visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(causal_mask, cmap='viridis')
    plt.colorbar()
    plt.title(f'Causal Mask Matrix (MSK version) - {dataset_name}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(mask_filename)
    plt.close()
    
    # Save evaluation results
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'causal_mask_sparsity': (causal_mask > 0.5).mean(),
        'target_idx': target_idx,
        'model_config': {
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'share_embedding': True
        },
        'dataset': dataset_name,
        'prefix': prefix
    }
    
    logger.info(f"\nFinal results summary:")
    if is_regression:
        logger.info(f"Validation MSE: {val_metrics['mse']:.4f}")
        logger.info(f"Validation R²: {val_metrics['r2']:.4f}")
        logger.info(f"Test MSE (final evaluation): {test_metrics['mse']:.4f}")
        logger.info(f"Test R² (final evaluation): {test_metrics['r2']:.4f}")
    else:
        logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Test Accuracy (final evaluation): {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC (final evaluation): {test_metrics['auc']:.4f}")
    logger.info(f"Causal mask sparsity: {(causal_mask > 0.5).mean():.4f}")
    
    # Save final results to file
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write(f"Causal Attention MSK Model Evaluation Results - {dataset_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Experiment prefix: {prefix if prefix else 'None'}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Dataset splitting method:\n")
        f.write(f"  1. Split training and test sets in 8:2 ratio\n")
        f.write(f"  2. Randomly take 20% of training set as validation set\n\n")
        f.write(f"Original dataset size: {total_size}\n")
        f.write(f"Training set size: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)\n")
        f.write(f"Validation set size: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)\n")
        f.write(f"Test set size: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)\n\n")
        
        f.write("Training set performance:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nValidation set performance:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nTest set performance (final evaluation):\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nCausal mask sparsity: {(causal_mask > 0.5).mean():.4f}\n")
    
    logger.info(f"\nDetailed results saved to: {results_filename}")
    
    return results

if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Causal attention MSK test')
    
    # Basic parameters
    parser.add_argument('--prefix', type=str, default='', 
                       help='Prefix for result files to distinguish different experiments (default: no prefix)')
    parser.add_argument('--dataset', type=str, default='housesale',
                       choices=['creditcard', 'adult', 'cardio', 'diamonds', 'elevator', 'housesale', 'crime', 'meps'],
                       help='Choose dataset: creditcard, adult, cardio, diamonds, elevator, housesale, crime, meps (default: housesale)')
    parser.add_argument('--gpu_id', type=int, default=-1, 
                       help='Specify GPU ID, -1 for default GPU, -2 for forcing CPU (default: -1)')
    
    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=64,
                       help='Model dimension (default: 64)')
    parser.add_argument('--num_heads', type=int, default=4, 
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=1, 
                       help='Number of Transformer layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd'],
                       help='Optimizer choice: adam, sgd (default: adam)')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', 
                       choices=['reduce_on_plateau', 'cosine'],
                       help='Learning rate scheduler: reduce_on_plateau, cosine (default: reduce_on_plateau)')
    
    # Loss weight parameters
    parser.add_argument('--alpha', type=float, default= 1,
                       help='Reconstruction loss weight (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.001,
                       help='Sparse and DAG loss weight (default: 0.001)')
    parser.add_argument('--regression_weight', type=float, default=5.0,
                       help='Regression task loss weight, to balance the magnitude difference between regression and classification tasks (default: 5.0)')
    
    # Update strategy parameters
    parser.add_argument('--recon_update_strategy', nargs=3, type=int, default=[2, 10, 1],
                       help='Reconstruction link update strategy (num interval offset) (default: [2, 10, 1])')
    parser.add_argument('--pred_update_strategy', nargs=3, type=int, default=[8, 10, 0],
                       help='Prediction link update strategy (num interval offset) (default: [8, 10, 0])')
    
    # Storage parameters
    parser.add_argument('--binary_mask_path', type=str, default=None,
                       help='File path to store binary_adj, if None, do not store (default: None)')

    parser.add_argument('--mask_threshold', type=float, default=0.5,
                       help='Threshold to binarize the causal mask for hierarchy/distance (default: 0.5)')
    
    # Model source parameters
    parser.add_argument('--model_source', type=str, default='models.causal_attention_msk_model',
                       help='Model source module path, e.g., models.causal_attention_msk_model (default: models.causal_attention_msk_model)')
    
    # Training function source parameters
    parser.add_argument('--train_source', type=str, default='utils.train_msk_utils',
                       help='Training function source module path, e.g., utils.train_msk_utils (default: utils.train_msk_utils)')
    
    # Global random seed parameter
    parser.add_argument('--seed', type=int, default=42, help='Global random seed (default: 42)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    print("=" * 60)
    print("Causal Attention MSK Model Experiment Configuration")
    print("=" * 60)
    print(f"Experiment prefix: {args.prefix if args.prefix else 'None'}")
    print(f"Dataset: {args.dataset}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Model source: {args.model_source}")
    print(f"Training function source: {args.train_source}")
    print("-" * 30)
    print("Model Architecture Parameters:")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Number of attention heads: {args.num_heads}")
    print(f"  Number of Transformer layers: {args.num_layers}")
    print(f"  Dropout rate: {args.dropout}")
    print("-" * 30)
    print("Training Parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Number of training epochs: {args.num_epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Batch size: {args.batch_size}")
    print("-" * 30)
    print("Optimizer Parameters:")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Scheduler: {args.scheduler}")
    print("-" * 30)
    print("Loss weight parameters:")
    print(f"  Reconstruction loss weight (alpha): {args.alpha}")
    print(f"  Sparse and DAG loss weight (beta): {args.beta}")
    print(f"  Regression task loss weight: {args.regression_weight}")
    print("-" * 30)
    print("Update strategy parameters:")
    print(f"  Reconstruction link update strategy: {args.recon_update_strategy}")
    print(f"  Prediction link update strategy: {args.pred_update_strategy}")
    print("=" * 60)
    
    # Run test
    seed = args.seed
    results = test_causal_attention_msk_model(
        prefix=args.prefix, 
        dataset=args.dataset, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers,
        gpu_id=args.gpu_id,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        alpha=args.alpha,
        beta=args.beta,
        regression_weight=args.regression_weight,
        recon_update_strategy=tuple(args.recon_update_strategy),
        pred_update_strategy=tuple(args.pred_update_strategy),
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        d_model=args.d_model,
        dropout=args.dropout,
        batch_size=args.batch_size,
        binary_mask_path=args.binary_mask_path,
        mask_threshold=args.mask_threshold,
        model_source=args.model_source,
        train_source=args.train_source,
        seed=seed
    )
    print("Causal Attention MSK Model test completed!")