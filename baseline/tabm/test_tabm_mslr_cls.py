import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import logging
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.mslr_loader import load_mslr_cls

import rtdl_num_embeddings
import tabm


class MSLRTabMDataset(Dataset):
    """MSLR-WEB10K: 136 continuous features, binary classification target."""

    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.X[idx]
        x_cat = np.array([], dtype=np.float32)  # no categorical features
        return (
            torch.tensor(x_num),
            torch.tensor(x_cat),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def test_tabm_mslr_cls(
    max_samples=100000,
    threshold=0,
    k=32,
    n_blocks=3,
    d_block=512,
    d_embedding=16,
    n_bins=48,
    batch_size=256,
    epochs=30,
    lr=2e-3,
    weight_decay=3e-4,
    patience=16,
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

    n_num_features = X.shape[1]

    # Quantile transform for numerical features (same as the regression TabM script)
    noise = np.random.default_rng(seed).normal(0.0, 1e-5, X_train.shape).astype(np.float32)
    preprocessing = QuantileTransformer(
        n_quantiles=max(min(len(X_train) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(X_train.astype(np.float32) + noise)

    X_train = preprocessing.transform(X_train).astype(np.float32)
    X_val = preprocessing.transform(X_val).astype(np.float32)
    X_test = preprocessing.transform(X_test).astype(np.float32)

    # y is already binary 0/1, no standardization needed
    train_ds = MSLRTabMDataset(X_train, y_train)
    val_ds = MSLRTabMDataset(X_val, y_val)
    test_ds = MSLRTabMDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Piecewise-linear embeddings for numerical features
    bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(X_train), n_bins=n_bins)
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bins, d_embedding=d_embedding, activation=False, version='B',
    )

    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_block,
        k=k,
        num_embeddings=num_embeddings,
    ).to(device)
    logger.info(f'TabM parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_loss_fn = nn.BCEWithLogitsLoss()

    def forward_batch(x_num):
        return model(x_num.to(device), None).squeeze(-1).float()

    def loss_on_batch(y_pred, y_true):
        y_pred = y_pred.flatten(0, 1)  # [B*k] logits
        y_true = y_true.repeat_interleave(model.backbone.k)
        return base_loss_fn(y_pred, y_true)

    @torch.inference_mode()
    def predict_loader(loader):
        model.eval()
        labels, preds = [], []
        for x_num, x_cat, y in loader:
            y_pred = torch.sigmoid(forward_batch(x_num)).mean(1)  # mean over k, then sigmoid
            labels.append(y.cpu().numpy())
            preds.append(y_pred.cpu().numpy())
        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        return y_true, y_pred

    def metrics_from_loader(loader):
        y_true, y_pred = predict_loader(loader)
        acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred)
        return acc, auc

    best_val_auc = -float('inf')
    best_state = None
    remaining_patience = patience

    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = forward_batch(x_num)
            loss = loss_on_batch(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_acc, val_auc = metrics_from_loader(val_loader)
        logger.info(f'Epoch {epoch + 1}: val_acc={val_acc:.4f}, val_auc={val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                logger.info(f'Early stopping @ epoch {epoch + 1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_auc = metrics_from_loader(test_loader)
    logger.info(f'MSLR TabM: Test ACC = {test_acc:.4f}, Test AUC = {test_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TabM on MSLR-WEB10K (binary classification)')
    parser.add_argument('--max_samples', type=int, default=100000, help='0 = full train docs (~723k)')
    parser.add_argument('--threshold', type=int, default=0,
                        help='relevance > threshold => positive (default 0)')
    parser.add_argument('--k', type=int, default=32, help='Number of ensemble sub-models')
    parser.add_argument('--n_blocks', type=int, default=3, help='Number of MLP blocks')
    parser.add_argument('--d_block', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--d_embedding', type=int, default=16, help='Numerical embedding dimension')
    parser.add_argument('--n_bins', type=int, default=48, help='Piecewise-linear embedding bins')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_tabm_mslr_cls(
        max_samples=args.max_samples,
        threshold=args.threshold,
        k=args.k,
        n_blocks=args.n_blocks,
        d_block=args.d_block,
        d_embedding=args.d_embedding,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        gpu_id=args.gpu_id,
        num_workers=args.num_workers,
        seed=args.seed,
    )
