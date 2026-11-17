import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import logging
import time
import sys
import os
import argparse
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.baseline_datasets import (
    load_creditcard_baseline_dataset,
    load_adult_baseline_dataset,
    load_cardio_baseline_dataset,
    load_diamonds_baseline_dataset,
    load_elevator_baseline_dataset,
    load_housesale_baseline_dataset,
    load_crime_baseline_dataset,
    load_meps_baseline_dataset,
)

import rtdl_num_embeddings
import tabm

DATASET_LOADERS = {
    'creditcard': load_creditcard_baseline_dataset,
    'adult': load_adult_baseline_dataset,
    'cardio': load_cardio_baseline_dataset,
    'cardiovascular': load_cardio_baseline_dataset,
    'diamonds': load_diamonds_baseline_dataset,
    'elevator': load_elevator_baseline_dataset,
    'housesale': load_housesale_baseline_dataset,
    'crime': load_crime_baseline_dataset,
    'meps': load_meps_baseline_dataset,
}


def convert_dataset_to_numpy(dataset, v, target_idx):
    all_data, all_targets = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_data.append(x.numpy())
        all_targets.append(y.item())
    return np.array(all_data), np.array(all_targets)


class TabMBatchDataset(Dataset):
    def __init__(self, data, targets, cat_idx, cont_idx, target_idx):
        self.data = data
        self.targets = targets
        self.cat_idx = cat_idx
        self.cont_idx = cont_idx
        self.target_idx = target_idx

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        if len(self.cat_idx) > 0:
            x_cat = x[self.cat_idx].astype(np.int64)
        else:
            x_cat = np.zeros(1, dtype=np.int64)
        if len(self.cont_idx) > 0:
            x_num = x[self.cont_idx].astype(np.float32)
        else:
            x_num = np.array([], dtype=np.float32)
        y = self.targets[idx]
        return (
            torch.tensor(x_num),
            torch.tensor(x_cat),
            torch.tensor(y, dtype=torch.float32),
        )


def test_tabm_baseline(
    dataset='creditcard',
    k=32,
    n_blocks=3,
    d_block=512,
    d_embedding=16,
    n_bins=48,
    gpu_id=-1,
    batch_size=256,
    val_batch_size=1024,
    epochs=30,
    lr=2e-3,
    weight_decay=3e-4,
    patience=16,
    num_workers=4,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(
        f'cuda:{gpu_id}'
        if torch.cuda.is_available() and gpu_id >= 0
        else 'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )
    logger.info(f'Device: {device}')
    logger.info(
        f'Parameters: dataset={dataset}, k={k}, n_blocks={n_blocks}, '
        f'd_block={d_block}, batch_size={batch_size}, epochs={epochs}, lr={lr}'
    )

    key = dataset.lower()
    if key not in DATASET_LOADERS:
        raise ValueError(
            f'Unsupported dataset: {dataset}, choices: {sorted(set(DATASET_LOADERS))}'
        )

    dataset_obj, v, num_classes_dict, target_idx = DATASET_LOADERS[key]()
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)

    logger.info(f'Dataset: {key}, size: {len(dataset_obj)}')
    logger.info(f'Feature types: {v}, target index: {target_idx}')

    # === Split: 80% train+val, 20% test; then 20% of train+val as val ===
    from utils.split_utils import split_indices
    total_size = len(dataset_obj)
    train_idx, val_idx, test_idx = split_indices(total_size, seed=seed)
    train_dataset = torch.utils.data.Subset(dataset_obj, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(dataset_obj, val_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset_obj, test_idx.tolist())
    logger.info(f'Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}')

    # Feature indices — excluding target
    cont_idx_all = (v == 0).nonzero(as_tuple=True)[0].numpy()
    cat_idx_all = (v == 1).nonzero(as_tuple=True)[0].numpy()
    cont_idx = cont_idx_all[cont_idx_all != target_idx]
    cat_idx = cat_idx_all[cat_idx_all != target_idx]

    cat_uniques = [num_classes_dict[int(i)] for i in cat_idx] if len(cat_idx) > 0 else []
    is_regression = v[target_idx].item() == 0
    task_type = 'regression' if is_regression else 'classification'
    logger.info(f'Task: {task_type}, cont features: {len(cont_idx)}, cat features: {len(cat_idx)}')

    # Convert to numpy
    train_data, train_targets = convert_dataset_to_numpy(train_dataset, v, target_idx)
    val_data, val_targets = convert_dataset_to_numpy(val_dataset, v, target_idx)
    test_data, test_targets = convert_dataset_to_numpy(test_dataset, v, target_idx)

    train_ds = TabMBatchDataset(train_data, train_targets, cat_idx, cont_idx, target_idx)
    val_ds = TabMBatchDataset(val_data, val_targets, cat_idx, cont_idx, target_idx)
    test_ds = TabMBatchDataset(test_data, test_targets, cat_idx, cont_idx, target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    n_num_features = max(len(cont_idx), 0)
    n_cat_features = len(cat_idx)
    cat_cardinalities = cat_uniques

    # Quantile transform for numerical features (same as TabM example)
    if n_num_features > 0:
        noise = np.random.default_rng(seed).normal(0.0, 1e-5, train_data[:, cont_idx].shape).astype(np.float32)
        preprocessing = QuantileTransformer(
            n_quantiles=max(min(len(train_data) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
        ).fit(train_data[:, cont_idx].astype(np.float32) + noise)

        def transform_num(arr):
            out = arr.copy()
            out[:, cont_idx] = preprocessing.transform(arr[:, cont_idx].astype(np.float32))
            return out

        train_data = transform_num(train_data)
        val_data = transform_num(val_data)
        test_data = transform_num(test_data)
        train_ds.data = train_data
        val_ds.data = val_data
        test_ds.data = test_data

    # Standardize regression target
    regression_label_stats = None
    if is_regression:
        mean, std = float(train_targets.mean()), float(train_targets.std())
        if std < 1e-8:
            std = 1.0
        regression_label_stats = (mean, std)
        train_ds.targets = (train_targets - mean) / std
        val_ds.targets = (val_targets - mean) / std
        test_ds.targets = (test_targets - mean) / std

    # Piecewise-linear embeddings for numerical features
    num_embeddings = None
    if n_num_features > 0:
        bins = rtdl_num_embeddings.compute_bins(
            torch.as_tensor(train_data[:, cont_idx].astype(np.float32)),
            n_bins=n_bins,
        )
        num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, d_embedding=d_embedding, activation=False, version='B',
        )

    if n_num_features == 0 and not cat_cardinalities:
        raise ValueError('Dataset has no usable features')

    # Build TabM model
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_block,
        k=k,
        num_embeddings=num_embeddings,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'TabM parameters: {total_params:,} ({total_params * 4 / 1024**2:.2f} MB)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_loss_fn = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()

    def forward_batch(x_num, x_cat):
        x_num_d = x_num.to(device) if n_num_features > 0 else None
        x_cat_d = x_cat.to(device) if n_cat_features > 0 else None
        return model(x_num_d, x_cat_d).squeeze(-1).float()

    def loss_on_batch(y_pred, y_true):
        y_pred = y_pred.flatten(0, 1)
        y_true = y_true.repeat_interleave(model.backbone.k)
        return base_loss_fn(y_pred, y_true)

    @torch.inference_mode()
    def predict_loader(loader):
        model.eval()
        labels, preds = [], []
        for x_num, x_cat, y in loader:
            y_pred = forward_batch(x_num, x_cat)
            if is_regression:
                y_pred = y_pred.mean(1)
            else:
                y_pred = torch.sigmoid(y_pred).mean(1)
            labels.append(y.cpu().numpy())
            preds.append(y_pred.cpu().numpy())
        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        if is_regression and regression_label_stats is not None:
            mean, std = regression_label_stats
            y_pred = y_pred * std + mean
        return y_true, y_pred

    def metrics_from_loader(loader):
        y_true, y_pred = predict_loader(loader)
        if is_regression:
            mse = mean_squared_error(y_true, y_pred)
            return mse, r2_score(y_true, y_pred)
        probs = y_pred
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float('nan')
        return acc, auc

    best_val_metric = -float('inf')
    best_state = None
    remaining_patience = patience
    ckpt_name = f'best_tabm_{key}.pth'

    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = forward_batch(x_num, x_cat)
            loss = loss_on_batch(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_m1, val_m2 = metrics_from_loader(val_loader)
        score = val_m2  # R² for regression, AUC for classification

        if is_regression:
            logger.info(f'Epoch {epoch + 1}: val_mse={val_m1:.4f}, val_r2={val_m2:.4f}')
        else:
            logger.info(f'Epoch {epoch + 1}: val_acc={val_m1:.4f}, val_auc={val_m2:.4f}')

        if score > best_val_metric:
            best_val_metric = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                logger.info(f'Early stopping @ epoch {epoch + 1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_name)

    test_m1, test_m2 = metrics_from_loader(test_loader)
    if is_regression:
        logger.info(f'Test MSE: {test_m1:.4f}, Test R2: {test_m2:.4f}')
    else:
        logger.info(f'Test ACC: {test_m1:.4f}, Test AUC: {test_m2:.4f}')
    logger.info(f'Checkpoint saved: {ckpt_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TabM Baseline')
    parser.add_argument('--dataset', type=str, default='creditcard',
                        choices=sorted(set(DATASET_LOADERS.keys())),
                        help='Dataset name')
    parser.add_argument('--k', type=int, default=32, help='Number of ensemble sub-models')
    parser.add_argument('--n_blocks', type=int, default=3, help='Number of MLP blocks')
    parser.add_argument('--d_block', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--d_embedding', type=int, default=16, help='Numerical embedding dimension')
    parser.add_argument('--n_bins', type=int, default=48, help='Piecewise-linear embedding bins')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_tabm_baseline(
        dataset=args.dataset,
        k=args.k,
        n_blocks=args.n_blocks,
        d_block=args.d_block,
        d_embedding=args.d_embedding,
        n_bins=args.n_bins,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )
