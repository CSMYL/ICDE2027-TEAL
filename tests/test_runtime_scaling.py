import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.meps_dataset import load_meps_reconstruct_dataset
from models.causal_attention_msk_model import CausalAttentionMskModel
from models.mask_generators import SigmoidMaskGenerator
from utils.train_msk_utils import train_msk_model


def main():
    parser = argparse.ArgumentParser(description='Runtime vs feature count scaling experiment')
    parser.add_argument('--dataset', default='meps', choices=['meps', 'crime', 'mslr', 'creditcard'])
    parser.add_argument('--num_samples', type=int, default=5000, help='fixed sample count')
    parser.add_argument('--num_epochs', type=int, default=5, help='training epochs per ratio (first epoch is warmup, excluded from timing)')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')

    if args.dataset == 'meps':
        dataset_obj, v, _ = load_meps_reconstruct_dataset()
    elif args.dataset == 'mslr':
        from datasets.mslr_dataset import load_mslr_reconstruct_dataset
        # MSLR-WEB10K: load exactly the fixed number of documents (already flattened + standardized).
        # 136 continuous features, relevance is the regression target (last column).
        dataset_obj, v, _ = load_mslr_reconstruct_dataset(max_samples=args.num_samples)
    elif args.dataset == 'creditcard':
        from datasets.creditcard_dataset import load_creditcard_reconstruct_dataset
        # CreditCard: 30 continuous features + binary Class target (last column).
        # The runtime experiment treats every column as continuous, so the binary
        # target is handled as a regression target for timing purposes only.
        dataset_obj, v, _ = load_creditcard_reconstruct_dataset()
    else:  # crime
        from datasets.crime_dataset import load_crime_reconstruct_dataset
        dataset_obj, v, _ = load_crime_reconstruct_dataset()
    F = len(v) - 1  # number of features (excluding the target)

    # Convert to numpy (all columns, target last)
    rows = [dataset_obj[i][0].numpy() for i in range(len(dataset_obj))]
    X = np.asarray(rows, dtype=np.float32)

    # Fix the sample count
    if len(X) > args.num_samples:
        idx = np.random.choice(len(X), args.num_samples, replace=False)
        X = X[idx]
    print(f'Dataset: {args.dataset}, samples: {len(X)}, total features: {F}')

    def build_and_time(k, num_epochs):
        """Train a model on the first k features + target and return the pure per-epoch
        training time (averaged over epochs 1..num_epochs-1; the first epoch is warmup)."""
        X_sub = X[:, list(range(k)) + [F]]
        x_tensor = torch.tensor(X_sub, dtype=torch.float32)
        ds = TensorDataset(x_tensor, x_tensor.clone())
        train_loader = DataLoader(ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(ds, batch_size=128, shuffle=False)

        # Treat all features as continuous for simplicity (runtime is dominated by F, not feature type)
        v_sub = torch.zeros(k + 1, dtype=torch.long, device=device)
        mask_gen = SigmoidMaskGenerator(num_features=k + 1, initial_threshold=0.2, final_threshold=0.2).to(device)
        model = CausalAttentionMskModel(
            v=v_sub, num_classes_dict={}, d_model=32, num_heads=4, num_layers=1,
            dropout=0.1, share_embedding=True, mask_generator=mask_gen, target_idx=k,
        ).to(device)

        train_msk_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            v=v_sub, num_classes_dict={}, test_loader=test_loader,
            prediction_idx=k, learning_rate=1e-4, weight_decay=1e-5,
            num_epochs=num_epochs, patience=10, alpha=1.0, beta=0.001,
            device=device, save_dir=None,
        )
        times = getattr(model, 'epoch_train_times', None)
        if times and len(times) > 1:
            return float(np.mean(times[1:]))
        return float(times[0]) if times else float('nan')

    print(f'{"ratio":>7} {"n_feat":>7} {"per_epoch_s":>12}')
    for ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = max(1, int(ratio * F))
        per_epoch = build_and_time(k, args.num_epochs)
        print(f'{ratio:>7.2f} {k:>7d} {per_epoch:>12.3f}')


if __name__ == '__main__':
    main()
