import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mslr_dataset import load_mslr_reconstruct_dataset_cls
from models.causal_attention_msk_model import CausalAttentionMskModel
from models.mask_generators import SigmoidMaskGenerator
from utils.train_msk_utils import train_msk_model, evaluate_test_set


def evaluate_multiclass(model, loader, target_idx, device):
    """5-class evaluation: accuracy + macro-F1."""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, prediction, _ = model(data, target_idx=target_idx, task_type='prediction')
            pred = prediction.squeeze(-1)
            pred_class = pred.argmax(dim=-1) if pred.dim() > 1 else (pred > 0.5).long().view(-1)
            all_preds.append(pred_class.cpu().numpy())
            all_targets.append(target[:, target_idx].cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


def main():
    parser = argparse.ArgumentParser(
        description='CAIM on MSLR-WEB10K as classification (binary by default)'
    )
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='subsample documents (0 = full ~723k train docs)')
    parser.add_argument('--multiclass', action='store_true',
                        help='use 5-class relevance (0-4) instead of binary (default)')
    parser.add_argument('--threshold', type=int, default=0,
                        help='binary threshold: relevance > threshold => relevant (default 0)')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1.0, help='reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=0.001, help='sparsity + DAG loss weight')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu'
    )

    binary = not args.multiclass
    max_samples = args.max_samples if args.max_samples > 0 else None
    dataset_obj, v, num_classes_dict = load_mslr_reconstruct_dataset_cls(
        max_samples=max_samples, binary=binary, threshold=args.threshold
    )
    v = torch.tensor(v).to(device)
    target_idx = len(v) - 1
    task_desc = f'binary (relevance>{args.threshold})' if binary else '5-class (relevance 0-4)'
    print(f'MSLR-cls: {len(dataset_obj)} samples, {len(v)} columns, '
          f'{num_classes_dict[target_idx]} classes, task={task_desc}')

    # Unified split: deterministic random shuffle -> 80/20 train-val/test -> 80/20 train/val
    from utils.split_utils import split_indices
    total = len(dataset_obj)
    train_idx, val_idx, test_idx = split_indices(total, seed=args.seed)
    train = Subset(dataset_obj, train_idx.tolist())
    val = Subset(dataset_obj, val_idx.tolist())
    test = Subset(dataset_obj, test_idx.tolist())

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)}')

    mask_gen = SigmoidMaskGenerator(num_features=len(v), initial_threshold=0.2, final_threshold=0.2).to(device)
    model = CausalAttentionMskModel(
        v=v, num_classes_dict=num_classes_dict, d_model=args.d_model,
        num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout,
        share_embedding=True, mask_generator=mask_gen, target_idx=target_idx,
    ).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    model = train_msk_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        v=v, num_classes_dict=num_classes_dict, test_loader=test_loader,
        prediction_idx=target_idx, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        num_epochs=args.num_epochs, patience=args.patience, alpha=args.alpha, beta=args.beta,
        device=device, save_dir=None,
    )

    if binary:
        results = evaluate_test_set(model, test_loader, target_idx, device, v)
        print(f'MSLR (binary, relevance>{args.threshold}) Test ACC: {results["acc"]:.4f}, '
              f'Test AUC: {results["auc"]:.4f}')
    else:
        acc, macro_f1 = evaluate_multiclass(model, test_loader, target_idx, device)
        print(f'MSLR (5-class) Test ACC: {acc:.4f}, Macro-F1: {macro_f1:.4f}')


if __name__ == '__main__':
    main()
