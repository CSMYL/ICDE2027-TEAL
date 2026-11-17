import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tab_transformer_pytorch import TabTransformer
import logging
import time
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.mslr_loader import load_mslr_cls


class MSLRTabDataset(Dataset):
    """MSLR-WEB10K: 136 continuous features, binary classification target."""

    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_categ = np.zeros(1, dtype=np.int64)  # dummy categorical (TabTransformer needs >=1)
        x_cont = self.X[idx]
        return (
            torch.tensor(x_categ),
            torch.tensor(x_cont),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def test_tabtransformer_mslr_cls(
    max_samples=100000,
    threshold=0,
    d_token=32,
    n_heads=4,
    n_blocks=6,
    batch_size=128,
    epochs=30,
    lr=1e-3,
    gpu_id=-1,
    num_workers=4,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device(
        f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    logger.info(f'Device: {device}')

    X, y = load_mslr_cls(
        max_samples=max_samples if max_samples > 0 else None,
        binary=True, threshold=threshold,
    )
    logger.info(f'MSLR: X {X.shape}, y {y.shape}, positive rate {y.mean():.3f}')

    # Unified split: deterministic random shuffle -> 80/20 train-val/test -> 80/20 train/val
    from utils.split_utils import split_indices
    train_idx, val_idx, test_idx = split_indices(len(X), seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    logger.info(f'Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}')

    train_ds = MSLRTabDataset(X_train, y_train)
    val_ds = MSLRTabDataset(X_val, y_val)
    test_ds = MSLRTabDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_cont = X.shape[1]

    # continuous mean/std (X is already standardized, so ~0/~1)
    cont_mean = np.mean(X_train, axis=0)
    cont_std = np.std(X_train, axis=0)
    cont_std = np.where(cont_std == 0, 1.0, cont_std)
    cont_mean_std = torch.tensor(np.stack([cont_mean, cont_std], axis=1), dtype=torch.float32)

    model = TabTransformer(
        categories=(2,),       # dummy categorical token
        num_continuous=num_cont,
        dim=d_token,
        dim_out=1,
        depth=n_blocks,
        heads=n_heads,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU(),
        continuous_mean_std=cont_mean_std,
    ).to(device)
    logger.info(f'TabTransformer parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def evaluate(loader):
        model.eval()
        labels, probs = [], []
        with torch.no_grad():
            for x_categ, x_cont, y in loader:
                x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
                p = torch.sigmoid(model(x_categ, x_cont).squeeze(-1))
                labels.append(y.cpu().numpy())
                probs.append(p.cpu().numpy())
        y_true = np.concatenate(labels)
        y_prob = np.concatenate(probs)
        acc = accuracy_score(y_true, (y_prob > 0.5).astype(int))
        auc = roc_auc_score(y_true, y_prob)
        return acc, auc

    best_auc = -float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for x_categ, x_cont, y in train_loader:
            x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_categ, x_cont).squeeze(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        logger.info(f'Epoch {epoch + 1}/{epochs} time: {time.time() - t0:.2f}s')

        val_acc, val_auc = evaluate(val_loader)
        logger.info(f'Epoch {epoch + 1}: val_acc={val_acc:.4f}, val_auc={val_auc:.4f}')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_auc = evaluate(test_loader)
    logger.info(f'MSLR TabTransformer: Test ACC = {test_acc:.4f}, Test AUC = {test_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TabTransformer on MSLR-WEB10K (binary classification)')
    parser.add_argument('--max_samples', type=int, default=100000, help='0 = full train docs (~723k)')
    parser.add_argument('--threshold', type=int, default=0,
                        help='relevance > threshold => positive (default 0)')
    parser.add_argument('--d_token', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_tabtransformer_mslr_cls(
        max_samples=args.max_samples,
        threshold=args.threshold,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gpu_id=args.gpu_id,
        num_workers=args.num_workers,
        seed=args.seed,
    )
