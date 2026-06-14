#!/usr/bin/env python
"""Fine-tuning entry point for HSI classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 src 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.datasets import HSIDataset
from src.models.encoder import HSIEncoder
from src.train import FinetuneEngine
from src.utils import get_logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSI-MAE Fine-tuning")

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--labels-path", type=str, required=True)
    parser.add_argument("--mat-key", type=str, default="data")
    parser.add_argument("--mat-label-key", type=str, default="labels")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=16)

    # Model
    parser.add_argument(
        "--encoder-path", type=str, required=True, help="Path to pretrained encoder"
    )
    parser.add_argument("--bands", type=int, default=200)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument(
        "--mode", type=str, default="full", choices=["linear_probe", "full"]
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Comma-separated class names for visualization",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup
    set_seed(args.seed)
    logger = get_logger(log_file=args.log_file)
    logger.info("=== HSI-MAE Fine-tuning ===")

    # Load pretrained encoder
    encoder = HSIEncoder(
        bands=args.bands,
        dim=args.encoder_dim,
        num_layers=args.encoder_layers,
    )
    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        # Fallback chain: try healthy checkpoints before final_encoder.pt
        for candidate in ["best_encoder.pt", "last_encoder.pt", "final_encoder.pt"]:
            fallback = Path(args.save_dir) / candidate
            if fallback.exists():
                # Verify checkpoint is not corrupted (has NaN)
                ckpt = torch.load(fallback, map_location="cpu", weights_only=False)
                state = ckpt["encoder_state"]
                has_nan = (
                    torch.isnan(torch.cat([v.flatten() for v in state.values()]))
                    .any()
                    .item()
                )
                if not has_nan:
                    encoder_path = fallback
                    logger.info(f"Using healthy checkpoint: {encoder_path}")
                    break
                else:
                    logger.warning(
                        f"Checkpoint {fallback} contains NaN, trying next..."
                    )
        else:
            raise FileNotFoundError(
                f"Encoder not found: {args.encoder_path} (no healthy checkpoints available)"
            )
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state"])
    logger.info(f"Encoder loaded from {encoder_path}")

    # Datasets
    train_ds = HSIDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        mat_key=args.mat_key,
        mat_label_key=args.mat_label_key,
        split="train",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        unlabeled=False,
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
        unlabeled=False,
        to_chw=True,
    )
    test_ds = HSIDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        mat_key=args.mat_key,
        mat_label_key=args.mat_label_key,
        split="test",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        unlabeled=False,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | "
        f"Classes: {args.num_classes}"
    )

    # Engine
    class_names = args.class_names.split(",") if args.class_names else None
    engine = FinetuneEngine(
        encoder=encoder,
        num_classes=args.num_classes,
        mode=args.mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        seed=args.seed,
        visualize=True,
        class_names=class_names,
    )

    # Train
    engine.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs)

    # Final test evaluation
    logger.info("=== Test Evaluation ===")
    engine.model.load_state_dict(
        torch.load(
            Path(args.save_dir) / "best_classifier.pt",
            map_location=engine.device,
            weights_only=False,
        )["model_state"]
    )

    metrics = engine._evaluate(test_loader)
    logger.info(f"Test OA: {metrics['oa']:.2%}")
    logger.info(f"Test AA: {metrics['aa']:.2%}")
    logger.info(f"Test Kappa: {metrics['kappa']:.4f}")
    logger.info(f"Test F1-macro: {metrics['f1_macro']:.4f}")

    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
