import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
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

CLASSIFICATION = {'creditcard', 'cardio', 'adult'}


def dataset_to_dataframe(dataset_obj, num_features):
    """Convert a reconstruction dataset to a pandas DataFrame (columns c0..cF)."""
    rows = []
    for i in range(len(dataset_obj)):
        x, _ = dataset_obj[i]
        rows.append(x.numpy())
    X = np.asarray(rows, dtype=np.float32)
    return pd.DataFrame(X, columns=[f'c{i}' for i in range(num_features)])


def run_hyfd(df, max_lhs=None):
    """Run HyFD (via Desbordante) and return a binarized adjacency matrix [d, d].

    HyFD (Papenbrock & Naumann, SIGMOD 2016) discovers functional dependencies
    via a hybrid sampling + validation approach and scales far better than TANE.

    max_lhs limits the maximum left-hand-side size explored (None = no limit).
    Setting max_lhs=1 or 2 is strongly recommended for wide datasets (e.g., MEPS)
    to keep FD discovery tractable.
    """
    import desbordante

    algos = desbordante.fd.algorithms
    cls = getattr(algos, 'HyFD', None) or getattr(algos, 'Hyfd', None)
    if cls is None:
        raise ImportError(
            'HyFD not found in desbordante.fd.algorithms. Available: '
            + ', '.join(n for n in dir(algos) if not n.startswith('_'))
        )

    algo = cls()
    algo.load_data(table=df)
    if max_lhs is not None:
        try:
            algo.execute(max_lhs=max_lhs)
        except TypeError:
            print('[WARNING] installed desbordante HyFD does not accept max_lhs; running without limit')
            algo.execute()
    else:
        algo.execute()
    fds = algo.get_fds()

    d = df.shape[1]
    adj = np.zeros((d, d), dtype=np.float32)

    for fd in fds:
        lhs = list(fd.lhs_indices)
        rhs = int(fd.rhs_index)
        for l in lhs:
            adj[int(l), rhs] = 1.0

    np.fill_diagonal(adj, 0)
    return adj


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
    parser = argparse.ArgumentParser(description='HyFD dependency graph as fixed mask + CAIM prediction')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--max_lhs', type=int, default=None,
                        help='max LHS size for HyFD FD discovery (None = no limit; set 1 or 2 for wide datasets like meps)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Load data
    dataset_obj, v, num_classes_dict = DATASET_LOADERS[args.dataset]()
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)
    target_idx = len(v) - 1
    logger.info(f'Dataset: {args.dataset}, size: {len(dataset_obj)}, features: {len(v)}')

    # Run HyFD on the data (converted to a DataFrame)
    df = dataset_to_dataframe(dataset_obj, len(v))
    logger.info(f'Running HyFD on {args.dataset} (shape {df.shape})...')
    adj = run_hyfd(df, max_lhs=args.max_lhs)
    n_edges = int(adj.sum())
    logger.info(f'HyFD graph: {n_edges} edges (sparsity {n_edges / (len(v) ** 2):.4f})')

    # Split data (8:2 train/test, then 8:2 train/val)
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

    # CAIM with fixed HyFD mask
    mask_gen = FixedMaskGenerator(adj).to(device)
    model = CausalAttentionMskModel(
        v=v, num_classes_dict=num_classes_dict, d_model=args.d_model,
        num_heads=args.num_heads, num_layers=args.num_layers, dropout=0.1,
        share_embedding=True, mask_generator=mask_gen, target_idx=target_idx,
    ).to(device)

    logger.info('Training CAIM with fixed HyFD mask...')
    model = train_msk_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        v=v, num_classes_dict=num_classes_dict, test_loader=test_loader,
        prediction_idx=target_idx, learning_rate=args.learning_rate, weight_decay=1e-5,
        num_epochs=args.num_epochs, patience=args.patience, alpha=1.0, beta=0.001,
        device=device, save_dir=f'checkpoints_hyfd_{args.dataset}',
    )

    results = evaluate_test_set(model, test_loader, target_idx, device, v)
    if args.dataset in CLASSIFICATION:
        logger.info(f'HyFD mask ({args.dataset}): AUC = {results["auc"]:.4f}')
    else:
        rmse = float(np.sqrt(results['mse']))
        logger.info(f'HyFD mask ({args.dataset}): RMSE = {rmse:.4f}')


if __name__ == '__main__':
    main()
