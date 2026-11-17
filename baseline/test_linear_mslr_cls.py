import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
import time
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.mslr_loader import load_mslr_cls


def test_linear_mslr_cls(
    max_samples=100000,
    threshold=0,
    batch_size=512,
    epochs=100,
    lr=1e-3,
    lambda_l1=5e-2,
    gpu_id=-1,
    seed=42,
):
    """LogCause-style linear baseline on MSLR (logistic regression + L1).

    Note: the DoWhy causal-effect weighting is skipped here because MSLR features are
    anonymous (f0..f135) and running DoWhy over 136 features is impractical; the baseline
    therefore degrades to plain logistic regression with uniform L1 regularization.
    """
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
    logger.info(f'MSLR: X {X.shape}, positive rate {y.mean():.3f}')

    from utils.split_utils import split_indices
    train_idx, _, test_idx = split_indices(len(X), seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    model = nn.Linear(input_dim, 1).to(device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def l1_loss(lambda_l1):
        return lambda_l1 * torch.sum(torch.abs(model.weight))

    def evaluate(loader):
        model.eval()
        labels, probs = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p = torch.sigmoid(model(x).squeeze(-1))
                labels.append(y.cpu().numpy())
                probs.append(p.cpu().numpy())
        y_true = np.concatenate(labels)
        y_prob = np.concatenate(probs)
        return accuracy_score(y_true, (y_prob > 0.5).astype(int)), roc_auc_score(y_true, y_prob)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(-1), y) + l1_loss(lambda_l1)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch + 1}/{epochs} time: {time.time() - t0:.2f}s')

    test_acc, test_auc = evaluate(test_loader)
    logger.info(f'MSLR LogCause: Test ACC = {test_acc:.4f}, Test AUC = {test_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogCause (linear) on MSLR-WEB10K (binary classification)')
    parser.add_argument('--max_samples', type=int, default=100000, help='0 = full ~723k docs')
    parser.add_argument('--threshold', type=int, default=0,
                        help='relevance > threshold => positive (default 0)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_l1', type=float, default=5e-2)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_linear_mslr_cls(
        max_samples=args.max_samples,
        threshold=args.threshold,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )
