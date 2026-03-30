#!/usr/bin/env python
"""Pre-training entry point for HSI-MAE."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets import HSIDataset
from src.train import PretrainEngine
from src.utils import get_logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSI-MAE Pre-training")

    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to HSI .npy or .mat file")
    parser.add_argument("--labels-path", type=str, default=None, help="Path to labels file (optional)")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--mat-key", type=str, default="data")
    parser.add_argument("--mat-label-key", type=str, default="labels")

    # Model
    parser.add_argument("--bands", type=int, default=200, help="Number of spectral bands")
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup
    set_seed(args.seed)
    logger = get_logger(log_file=args.log_file)
    logger.info("=== HSI-MAE Pre-training ===")
    logger.info(f"Data: {args.data_path}")

    # Datasets
    train_ds = HSIDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        mat_key=args.mat_key,
        mat_label_key=args.mat_label_key,
        split="train",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        unlabeled=True,  # No labels needed for MAE
        to_chw=True,
    )
    val_ds = HSIDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        mat_key=args.mat_key,
        mat_label_key=args.mat_label_key,
        split="val",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        unlabeled=True,
        to_chw=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples | "
        f"Bands: {train_ds.num_bands}"
    )

    # Engine
    engine = PretrainEngine(
        bands=args.bands,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        decoder_dim=args.decoder_dim,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        seed=args.seed,
    )

    # Run
    engine.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs)

    # Save final
    final_path = Path(args.save_dir) / "final_encoder.pt"
    import torch
    torch.save({"encoder_state": engine.model.encoder.state_dict()}, final_path)
    logger.info(f"Final encoder saved: {final_path}")


if __name__ == "__main__":
    main()
