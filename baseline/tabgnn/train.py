#!/usr/bin/env python
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))

from data.CSVToGraphAdapter import CSVToGraphAdapter
from models.GNN.GCN import GCN
from utils import get_DGL_collator


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN model')

    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (must be registered in data/tabular_ds_info.json)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train split ratio (default 0.8)')
    parser.add_argument('--connect_keys', type=str, nargs='*', default=None,
                       help='Connect key list (categorical feature names), None for auto-detect or similarity')
    parser.add_argument('--max_neighbors', type=int, default=10,
                       help='Max neighbors per connect key (default 10)')

    parser.add_argument('--model', type=str, default='GCN',
                       choices=['GCN'],
                       help='Model type (default GCN)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension (default 64)')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of GNN layers (default 3)')
    parser.add_argument('--fcout_layers', type=int, nargs='+', default=[64, 32],
                       help='Output MLP layer sizes (default [64, 32])')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default 0.2)')

    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default 0.001)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (default cpu)')

    return parser.parse_args()


def get_metric_name(task_type):
    if task_type == 'binary classification' or task_type == 'multiclass classification':
        return 'AUC'
    elif task_type == 'regression':
        return 'RMSE'
    else:
        raise ValueError(f'Unsupported task type: {task_type}')


def evaluate(model, loader, criterion, task_type, device):
    model.eval()
    all_labels = []
    all_outputs = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            (bdgl, features, main_node_ids), labels = batch

            if bdgl.device != device:
                bdgl = bdgl.to(device)
            labels = labels.to(device)

            device_features = {}
            for node_type, node_features in features.items():
                if isinstance(node_features, tuple):
                    cat_feats, cont_feats = node_features
                    if isinstance(cat_feats, torch.Tensor) and cat_feats.device != device:
                        cat_feats = cat_feats.to(device)
                    if isinstance(cont_feats, torch.Tensor) and cont_feats.device != device:
                        cont_feats = cont_feats.to(device)
                    device_features[node_type] = (cat_feats, cont_feats)
                else:
                    device_features[node_type] = node_features

            input_data = (bdgl, device_features, main_node_ids)
            outputs = model(input_data)

            labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    avg_loss = total_loss / len(loader)

    if task_type == 'binary classification':
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        metric_value = roc_auc_score(all_labels, probs[:, 1])
    elif task_type == 'multiclass classification':
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        metric_value = roc_auc_score(all_labels, probs, multi_class='ovr')
    elif task_type == 'regression':
        predictions = all_outputs.squeeze()
        metric_value = np.sqrt(mean_squared_error(all_labels, predictions))
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

    return avg_loss, metric_value


def train_epoch(model, loader, criterion, optimizer, device, task_type='binary classification'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    batch_idx = 0
    for batch in loader:
        batch_idx += 1

        (bdgl, features, main_node_ids), labels = batch

        optimizer.zero_grad()

        if bdgl.device != device:
            bdgl = bdgl.to(device)
        labels = labels.to(device)

        device_features = {}
        for node_type, node_features in features.items():
            if isinstance(node_features, tuple):
                cat_feats, cont_feats = node_features
                if isinstance(cat_feats, torch.Tensor) and cat_feats.device != device:
                    cat_feats = cat_feats.to(device)
                if isinstance(cont_feats, torch.Tensor) and cont_feats.device != device:
                    cont_feats = cont_feats.to(device)
                device_features[node_type] = (cat_feats, cont_feats)
            else:
                device_features[node_type] = node_features

        input_data = (bdgl, device_features, main_node_ids)
        outputs = model(input_data)

        if task_type == 'regression':
            labels = labels.float()
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
        else:
            labels = labels.long()

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if task_type == 'regression':
            total += labels.size(0)
        else:
            if outputs.dim() > 1:
                predictions = outputs.argmax(dim=1)
            else:
                predictions = (outputs > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if batch_idx % 20 == 0:
            avg_loss = total_loss / batch_idx
            acc = correct / total if total > 0 else 0
            print(f'  Batch {batch_idx}/{len(loader)}: Loss={avg_loss:.4f}, Acc={acc:.4f}', flush=True)

    return total_loss / len(loader), correct / total if total > 0 else 0


def main():
    args = parse_args()

    print('=' * 60)
    print('GNN Model Training')
    print('=' * 60)
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Hidden dim: {args.hidden_dim}')
    print(f'GNN layers: {args.n_layers}')
    print('=' * 60)

    from data.utils import get_ds_info
    try:
        ds_info = get_ds_info(args.dataset)
        if 'processed' in ds_info and 'task' in ds_info['processed']:
            task_type = ds_info['processed']['task']
        elif 'meta' in ds_info and 'task' in ds_info['meta']:
            task_type = ds_info['meta']['task']
        else:
            raise KeyError('Cannot find task field')
        print(f'\nTask type: {task_type}')
    except Exception as e:
        print(f'Failed to load dataset info: {e}')
        import traceback
        traceback.print_exc()
        return

    encoders = {
        'NUMERIC': 'ScalarRobustScalerEnc',
        'CATEGORICAL': 'CategoricalOrdinalEnc'
    }

    print(f'\n1. Loading dataset: {args.dataset}')
    try:
        full_dataset = CSVToGraphAdapter(
            dataset_name=args.dataset,
            encoders=encoders,
            connect_keys=args.connect_keys,
            max_neighbors_per_key=args.max_neighbors
        )
        print(f'   Dataset loaded: {len(full_dataset)} samples')
        print(f'   Connect keys: {full_dataset.connect_keys}')
    except Exception as e:
        print(f'   Dataset loading failed: {e}')
        import traceback
        traceback.print_exc()
        return

    n_total = len(full_dataset)
    n_train = int(n_total * args.train_split)
    train_ids = np.arange(n_train)
    test_ids = np.arange(n_train, n_total)

    train_dataset = CSVToGraphAdapter(
        dataset_name=args.dataset,
        datapoint_ids=train_ids,
        encoders=encoders,
        connect_keys=args.connect_keys,
        max_neighbors_per_key=args.max_neighbors
    )
    test_dataset = CSVToGraphAdapter(
        dataset_name=args.dataset,
        datapoint_ids=test_ids,
        encoders=encoders,
        connect_keys=args.connect_keys,
        max_neighbors_per_key=args.max_neighbors
    )

    print(f'   Train set: {len(train_dataset)} samples')
    print(f'   Test set: {len(test_dataset)} samples')

    device = torch.device(args.device)

    print('\n2. Creating data loaders')
    try:
        print('   Creating collator...', flush=True)
        collator_device = device if device.type == 'cuda' else 'cpu'
        collator = get_DGL_collator(train_dataset.feature_encoders, train_dataset.db_info, device=collator_device)
        print(f'   Collator created (device={collator_device}), creating DataLoader...', flush=True)
        pin_memory = (device.type == 'cuda' and collator_device == 'cpu')
        print(f'   pin_memory={pin_memory} (collator_device={collator_device}, device={device})', flush=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0, pin_memory=pin_memory)
        print(f'   DataLoaders created: train batches={len(train_loader)}, test batches={len(test_loader)}', flush=True)
    except Exception as e:
        print(f'   DataLoader creation failed: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return

    print('\n3. Initializing model')
    try:
        if args.model == 'GCN':
            model = GCN(
                writer=None,
                dataset_name=args.dataset,
                feature_encoders=train_dataset.feature_encoders,
                hidden_dim=args.hidden_dim,
                init_model_class_name='TabMLP',
                init_model_kwargs={
                    'layer_sizes': [args.hidden_dim],
                    'max_emb_dim': 50,
                    'activation_class_name': 'GELU',
                    'activation_class_kwargs': {},
                    'norm_class_name': 'BatchNorm1d',
                    'norm_class_kwargs': {},
                    'one_hot_embeddings': False,
                    'drop_whole_embeddings': False,
                    'p_dropout': args.dropout * 0.5
                },
                n_layers=args.n_layers,
                activation_class_name='GELU',
                activation_class_kwargs={},
                norm_class_name='BatchNorm1d',
                norm_class_kwargs={},
                loss_class_name='CrossEntropyLoss' if task_type != 'regression' else 'MSELoss',
                loss_class_kwargs={},
                p_dropout=args.dropout,
                readout_class_name='GlobalAttentionPooling',
                readout_kwargs={
                    'n_layers': 2,
                    'act_name': 'GELU'
                },
                fcout_layer_sizes=args.fcout_layers,
                use_jknet=False,
                cat_fz_embedding=False
            )
        else:
            raise ValueError(f'Unsupported model: {args.model}')

        n_params = sum(p.numel() for p in model.parameters())
        print(f'   Model initialized')
        print(f'   Parameters: {n_params:,}')
    except Exception as e:
        print(f'   Model initialization failed: {e}')
        import traceback
        traceback.print_exc()
        return

    device = torch.device(args.device)
    model.to(device)

    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f'\n4. Training ({args.epochs} epochs)')
    metric_name = get_metric_name(task_type)

    best_test_metric = float('inf') if task_type == 'regression' else 0.0

    for epoch in range(1, args.epochs + 1):
        train_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, task_type)
        train_time = time.time() - train_start_time
        train_time_min = train_time / 60

        test_start_time = time.time()
        test_loss, test_metric = evaluate(model, test_loader, criterion, task_type, device)
        test_time = time.time() - test_start_time
        test_time_min = test_time / 60

        is_best = False
        if task_type == 'regression':
            if test_metric < best_test_metric:
                best_test_metric = test_metric
                is_best = True
        else:
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                is_best = True

        marker = ' *' if is_best else ''
        if task_type == 'regression':
            print(f'Epoch {epoch}/{args.epochs}: '
                  f'Train Loss: {train_loss:.4f} ({train_time_min:.2f}min) | '
                  f'Test Loss: {test_loss:.4f}, Test {metric_name}: {test_metric:.4f} ({test_time_min:.2f}min){marker}')
        else:
            print(f'Epoch {epoch}/{args.epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_time_min:.2f}min) | '
                  f'Test Loss: {test_loss:.4f}, Test {metric_name}: {test_metric:.4f} ({test_time_min:.2f}min){marker}')

    print('\n' + '=' * 60)
    print(f'Training complete!')
    print(f'Best test {metric_name}: {best_test_metric:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
