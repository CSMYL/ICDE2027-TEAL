import os
import sys
import argparse
import logging
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from torch.utils.data import DataLoader, Subset, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset
from datasets.cardio_dataset import load_cardio_reconstruct_dataset
from datasets.adult_dataset import load_adult_reconstruct_dataset
from datasets.diamonds_dataset import load_diamonds_reconstruct_dataset
from datasets.elevator_dataset import load_elevator_reconstruct_dataset
from datasets.housesale_dataset import load_housesale_reconstruct_dataset
from datasets.crime_dataset import load_crime_reconstruct_dataset
from datasets.meps_dataset import load_meps_reconstruct_dataset

from models.causal_attention_msk_model import CausalAttentionMskModel
from utils.train_msk_utils import train_msk_model, evaluate_test_set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_LOADERS = {
    'creditcard': load_creditcard_reconstruct_dataset,
    'cardio': load_cardio_reconstruct_dataset,
    'adult': load_adult_reconstruct_dataset,
    'diamonds': load_diamonds_reconstruct_dataset,
    'elevator': load_elevator_reconstruct_dataset,
    'housesale': load_housesale_reconstruct_dataset,
    'crime': load_crime_reconstruct_dataset,
    'meps': load_meps_reconstruct_dataset,
}

CLASSIFICATION_DATASETS = {'creditcard', 'cardio', 'adult'}


def run_notears(X, lambda1=0.1, max_iter=100):
    """Self-contained NOTEARS (linear, L2) returning a binarized adjacency [d, d].

    Implements the NOTEARS augmented-Lagrangian optimization directly (only numpy + scipy),
    so no external `notears` package is needed.
    """
    n, d = X.shape

    def _h(W):
        # acyclicity: h(W) = tr(exp(W o W)) - d
        return np.trace(slin.expm(np.square(W))) - d

    def _h_grad(W):
        # d/dW tr(exp(W o W)) = 2 * W o exp(W o W)^T
        return slin.expm(np.square(W)).T * W * 2.0

    def _obj(w, rho, alpha):
        W = w.reshape(d, d)
        R = X - X @ W
        loss = 0.5 / n * np.sum(R ** 2)          # l2 reconstruction loss
        G = -1.0 / n * X.T @ R                   # its gradient
        h = _h(W)
        f = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.sum(np.abs(W))
        g = G + (rho * h + alpha) * _h_grad(W) + lambda1 * np.sign(W)
        return f, g.reshape(-1)

    W_est = np.zeros((d, d))
    alpha, rho, h = 0.0, 1.0, np.inf
    bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        res = sopt.minimize(_obj, W_est.reshape(-1), args=(rho, alpha), jac=True,
                            method='L-BFGS-B', bounds=bnds)
        W_est = res.x.reshape(d, d)
        h_new = _h(W_est)
        if h_new > 0.25 * h:
            rho *= 10.0
        alpha += rho * h_new
        h = h_new
        if h <= 1e-8 or rho >= 1e16:
            break

    adj = (np.abs(W_est) > 0.3).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def dataset_to_numpy(dataset_obj, max_samples=None):
    """Convert a reconstruction dataset to a numpy matrix [n, d]."""
    X = []
    n = len(dataset_obj)
    if max_samples is not None and 0 < max_samples < n:
        idx = np.linspace(0, n - 1, max_samples).astype(int)
    else:
        idx = range(n)
    for i in idx:
        x, _ = dataset_obj[i]
        X.append(x.numpy())
    return np.array(X)


class FixedMaskGenerator(torch.nn.Module):
    """A mask generator that always returns a fixed (non-learnable) causal mask."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer('fixed_mask', torch.tensor(mask, dtype=torch.float32))

    def forward(self, mask_logits, batch_size):
        return self.fixed_mask.unsqueeze(0).expand(batch_size, -1, -1)

    def get_causal_mask(self, mask_logits):
        return self.fixed_mask

    def update_parameters(self):
        pass

    def update_parameters_for_early_stopping(self):
        pass

    def get_parameters(self):
        return {}

    def initialize_mask_logits(self, device='cpu'):
        return torch.zeros(1, 1, device=device)


def main():
    parser = argparse.ArgumentParser(description='NOTEARS mask export + CAIM with fixed NOTEARS mask')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument('--lambda1', type=float, default=0.1, help='NOTEARS L1 regularization')
    parser.add_argument('--max_iter', type=int, default=100, help='NOTEARS max iterations')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='subsample size for NOTEARS (0 = use full data; large datasets like creditcard/meps should subsample)')
    parser.add_argument('--force_rerun', action='store_true', help='re-run NOTEARS even if the exported .npy already exists')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # ---- Load dataset ----
    dataset_obj, v, num_classes_dict = DATASET_LOADERS[args.dataset]()
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)
    target_idx = len(v) - 1
    logger.info(f'Dataset: {args.dataset}, size: {len(dataset_obj)}, features: {len(v)}')

    # ---- Run NOTEARS (or reuse exported mask) ----
    adj_path = os.path.join(PROJECT_ROOT, 'raw_data', f'notears_adj_{args.dataset}.npy')
    if os.path.exists(adj_path) and not args.force_rerun:
        adj = np.load(adj_path)
        logger.info(f'Loaded existing NOTEARS mask from {adj_path}')
    else:
        X = dataset_to_numpy(dataset_obj, args.max_samples if args.max_samples > 0 else None)
        logger.info(f'Running NOTEARS on {args.dataset} (X shape: {X.shape}, lambda1={args.lambda1})...')
        adj = run_notears(X, lambda1=args.lambda1, max_iter=args.max_iter)
        np.save(adj_path, adj)
        logger.info(f'Exported NOTEARS mask to {adj_path}')

    n_edges = int(adj.sum())
    logger.info(f'NOTEARS graph: {n_edges} edges (sparsity {n_edges / (len(v) ** 2):.4f})')

    # ---- Split data (same as the main test script: 8:2 train/test, then 8:2 train/val) ----
    from utils.split_utils import split_indices
    total_size = len(dataset_obj)
    train_idx, val_idx, test_idx = split_indices(total_size, seed=args.seed)
    train_set = Subset(dataset_obj, train_idx.tolist())
    val_set = Subset(dataset_obj, val_idx.tolist())
    test_set = Subset(dataset_obj, test_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, drop_last=True)
    logger.info(f'Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}')

    # ---- CAIM with fixed NOTEARS mask ----
    mask_gen = FixedMaskGenerator(adj).to(device)
    model = CausalAttentionMskModel(
        v=v, num_classes_dict=num_classes_dict, d_model=args.d_model,
        num_heads=args.num_heads, num_layers=args.num_layers, dropout=0.1,
        share_embedding=True, mask_generator=mask_gen, target_idx=target_idx,
    ).to(device)

    logger.info('Training CAIM with fixed NOTEARS mask...')
    model = train_msk_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        v=v, num_classes_dict=num_classes_dict, test_loader=test_loader,
        prediction_idx=target_idx, learning_rate=args.learning_rate, weight_decay=1e-5,
        num_epochs=args.num_epochs, patience=args.patience, alpha=1.0, beta=0.001,
        device=device, save_dir=f'checkpoints_notears_{args.dataset}',
    )

    # ---- Evaluate ----
    results = evaluate_test_set(model, test_loader, target_idx, device, v)
    if args.dataset in CLASSIFICATION_DATASETS:
        logger.info(f'NOTEARS mask ({args.dataset}): AUC = {results["auc"]:.4f}')
    else:
        rmse = float(np.sqrt(results['mse']))
        logger.info(f'NOTEARS mask ({args.dataset}): RMSE = {rmse:.4f}')


if __name__ == '__main__':
    main()
