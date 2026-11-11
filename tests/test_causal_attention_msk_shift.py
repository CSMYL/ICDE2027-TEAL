#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from datasets.ood_shift_datasets import _load_shift_reconstruct_dataset
from models.mask_generators import SigmoidMaskGenerator


def get_model_class(model_source="models.causal_attention_msk_model"):
    """Dynamically import model class based on model_source (consistent with test_causal_attention_msk_model.py)."""
    try:
        module = __import__(model_source, fromlist=["CausalAttentionMskModel"])
        return getattr(module, "CausalAttentionMskModel")
    except ImportError as e:
        raise ImportError(f"Cannot import model module {model_source}: {e}")
    except AttributeError as e:
        raise AttributeError(f"CausalAttentionMskModel not found in module {model_source}: {e}")


def get_train_function(train_source="utils.train_msk_utils"):
    """Dynamically import training function based on train_source (consistent with test_causal_attention_msk_model.py)."""
    try:
        module = __import__(train_source, fromlist=["train_msk_model"])
        return getattr(module, "train_msk_model")
    except ImportError as e:
        raise ImportError(f"Cannot import training module {train_source}: {e}")
    except AttributeError as e:
        raise AttributeError(f"train_msk_model not found in module {train_source}: {e}")


def build_shift_dataset(shift_name: str):
    """
    Load shift1/2/3 with corresponding train_expX/test_expX size info.
    Returns:
        dataset_obj: entire shift data (train+test concatenated)
        v: feature type vector
        num_classes_dict: cardinality of categorical features
        train_size: number of OOD train samples (used for indexing)
    """
    assert shift_name in ("shift1", "shift2", "shift3")
    exp_id = int(shift_name[-1])

    base_dir = os.path.join(os.path.dirname(__file__), "../raw_data/diamonds_ood_experiments")
    base_dir = os.path.abspath(base_dir)

    shift_csv = os.path.join(base_dir, f"{shift_name}.csv")
    train_csv = os.path.join(base_dir, f"train_exp{exp_id}.csv")
    test_csv = os.path.join(base_dir, f"test_exp{exp_id}.csv")

    if not os.path.exists(shift_csv):
        raise FileNotFoundError(
            f"{shift_csv} not found. Run generate_ood_splits.py and generate_shift_datasets.py first."
        )
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError(f"Missing OOD split files: {train_csv} or {test_csv}")

    dataset_obj, v, num_classes_dict = _load_shift_reconstruct_dataset(shift_csv, shuffle=False)

    import pandas as pd

    n_train = len(pd.read_csv(train_csv))
    n_total = len(dataset_obj)
    if n_total != 2 * n_train:
        raise ValueError(
            f"{shift_name}: expected shift size = 2 * |train_exp{exp_id}|, "
            f"but got shift={n_total}, train_exp={n_train}"
        )

    return dataset_obj, v, num_classes_dict, n_train


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load shift dataset
    dataset_obj, v, num_classes_dict, ood_train_size = build_shift_dataset(args.dataset)
    total_size = len(dataset_obj)
    ood_test_size = total_size - ood_train_size

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Total samples: {total_size} (OOD train={ood_train_size}, OOD test={ood_test_size})")
    logger.info(f"Feature type vector v: {v}")
    logger.info(f"v shape: {v.shape}")
    logger.info(f"Categorical feature cardinalities: {num_classes_dict}")

    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)

    sample_x, sample_y = dataset_obj[0]
    logger.info(f"Sample input shape: {sample_x.shape}")
    logger.info(f"Sample target shape: {sample_y.shape}")

    target_idx = len(v) - 1
    logger.info(f"Prediction target index (last column): {target_idx}")

    # Split OOD Train further into train/val; keep OOD Test unchanged
    train_val_indices = list(range(0, ood_train_size))
    test_indices = list(range(ood_train_size, total_size))

    ood_train_dataset = Subset(dataset_obj, train_val_indices)
    test_dataset = Subset(dataset_obj, test_indices)

    train_val_size = len(ood_train_dataset)
    val_size = int(args.val_ratio * train_val_size)
    train_size = train_val_size - val_size

    train_dataset, val_dataset = random_split(
        ood_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"OOD Train total: {train_val_size}")
    logger.info(f"  -> Train: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    logger.info(f"  -> Val:   {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    logger.info(f"OOD Test:  {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("\nInitializing mask generator...")
    mask_generator = SigmoidMaskGenerator(
        num_features=len(v),
        initial_threshold=0.2,
        final_threshold=0.2,
        threshold_multiplier=1.1,
    ).to(device)

    logger.info(f"\nImporting model class from {args.model_source}...")
    ModelClass = get_model_class(args.model_source)

    logger.info(f"\nImporting training function from {args.train_source}...")
    train_function = get_train_function(args.train_source)

    logger.info("\nInitializing causal attention MSK model...")
    model = ModelClass(
        v=v,
        num_classes_dict=num_classes_dict,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        share_embedding=True,
        mask_generator=mask_generator,
        target_idx=target_idx,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("\nStarting training (based on shift OOD split)...")
    trained_model = train_function(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        v=v,
        num_classes_dict=num_classes_dict,
        test_loader=test_loader,
        prediction_idx=target_idx,
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
        device=device,
        save_dir=f"checkpoints_causal_msk_{args.prefix}" if args.prefix else "checkpoints_causal_msk",
        log_interval=100,
    )

    logger.info("\nStarting evaluation on OOD Test...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Attention MSK OOD Shift Experiment")

    parser.add_argument(
        "--dataset",
        type=str,
        default="shift1",
        choices=["shift1", "shift2", "shift3"],
        help="OOD shift dataset: shift1(Cut), shift2(Color), shift3(Simpson) (default: shift1)",
    )
    parser.add_argument("--prefix", type=str, default="", help="Results file prefix")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID (-1 for default, -2 to force CPU)")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio within OOD Train (default 0.2)",
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--regression_weight", type=float, default=5.0)
    parser.add_argument(
        "--recon_update_strategy",
        type=int,
        nargs=3,
        default=[2, 10, 1],
        metavar=("NUM", "INTERVAL", "OFFSET"),
    )
    parser.add_argument(
        "--pred_update_strategy",
        type=int,
        nargs=3,
        default=[8, 10, 0],
        metavar=("NUM", "INTERVAL", "OFFSET"),
    )

    parser.add_argument(
        "--model_source",
        type=str,
        default="models.causal_attention_msk_model",
        help="Python module path for model class (consistent with main script)",
    )
    parser.add_argument(
        "--train_source",
        type=str,
        default="utils.train_msk_utils",
        help="Python module path for training function (consistent with main script)",
    )

    args = parser.parse_args()
    main(args)
