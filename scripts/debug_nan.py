#!/usr/bin/env python
"""Debug NaN issue in fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.datasets import HSIDataset
from src.models.classifier import HSIFineTuner
from src.models.encoder import HSIEncoder


def check_tensor(name: str, t: torch.Tensor) -> bool:
    """Check if tensor has NaN or Inf."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(
        f"  {name}: shape={t.shape}, dtype={t.dtype}, device={t.device}, "
        f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
        f"nan={has_nan}, inf={has_inf}"
    )
    return has_nan or has_inf


def main():
    data_path = "data/indian-pines/indian_pines.npy"
    labels_path = "data/indian-pines/indian_pines_gt.npy"
    encoder_path = "checkpoints/final_encoder.pt"

    print("=== Loading encoder from checkpoint ===")
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    print(f"  Keys in checkpoint: {list(ckpt.keys())}")

    encoder_state = ckpt["encoder_state"]
    has_nan = (
        torch.isnan(torch.cat([v.flatten() for v in encoder_state.values()]))
        .any()
        .item()
    )
    print(f"  Encoder state has NaN: {has_nan}")

    print("\n=== Creating encoder ===")
    encoder = HSIEncoder(bands=200, dim=256, num_layers=4)
    print(f"  Encoder output dim: {encoder.get_output_dim()}")

    print("\n=== Loading encoder state ===")
    encoder.load_state_dict(encoder_state)
    print("  Load successful")

    print("\n=== Checking encoder params ===")
    for name, param in encoder.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  NaN/Inf in encoder.{name}: {param.shape}")

    print("\n=== Creating fine-tuner ===")
    model = HSIFineTuner(encoder=encoder, num_classes=16, classifier_variant="mlp")
    print("  Model created")

    print("\n=== Moving model to CUDA ===")
    if torch.cuda.is_available():
        model = model.cuda()
        print("  Model on CUDA")

        # Check classifier params for NaN
        print("\n=== Checking classifier params on CUDA ===")
        for name, param in model.classifier.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  NaN/Inf in classifier.{name}: {param.shape}")
            else:
                print(
                    f"  OK: classifier.{name}: {param.shape}, "
                    f"min={param.min().item():.4f}, max={param.max().item():.4f}"
                )

        print("\n=== Loading a batch from dataset ===")
        ds = HSIDataset(
            data_path=data_path,
            labels_path=labels_path,
            split="train",
            train_ratio=0.8,
            val_ratio=0.1,
            unlabeled=False,
            to_chw=True,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        x, y = next(iter(loader))
        print(f"  Input x: shape={x.shape}, dtype={x.dtype}")
        print(f"  Labels y: shape={y.shape}, dtype={y.dtype}")

        check_tensor("Input x", x)

        print("\n=== Moving data to CUDA ===")
        x = x.cuda()
        y = y.cuda()
        check_tensor("Input x on CUDA", x)

        print("\n=== Forward pass through encoder ===")
        with torch.no_grad():
            features = encoder(x)
            check_tensor("Encoder features", features)

        print("\n=== Forward pass through full model ===")
        with torch.no_grad():
            logits = model(x)
            check_tensor("Logits", logits)

        print("\n=== Backward pass (test gradient) ===")
        model.train()
        x.requires_grad = True
        logits = model(x)
        print(f"  Logits shape: {logits.shape}")

        loss = torch.nn.functional.cross_entropy(
            logits.squeeze(-1).squeeze(-1), y.squeeze()
        )
        print(f"  Loss: {loss.item():.4f}")

        if torch.isnan(loss):
            print("  WARNING: Loss is NaN!")
        else:
            print("  Loss is valid")

        loss.backward()
        print("\n=== Checking gradients ===")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  NaN/Inf gradient in {name}")
                else:
                    print(
                        f"  OK: gradient in {name}, grad_norm={param.grad.norm().item():.4f}"
                    )
            else:
                print(f"  No gradient: {name}")

    else:
        print("CUDA not available, testing on CPU")

        # Test on CPU
        print("\n=== Loading a batch from dataset ===")
        ds = HSIDataset(
            data_path=data_path,
            labels_path=labels_path,
            split="train",
            train_ratio=0.8,
            val_ratio=0.1,
            unlabeled=False,
            to_chw=True,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        x, y = next(iter(loader))
        print(f"  Input x: shape={x.shape}, dtype={x.dtype}")
        check_tensor("Input x", x)

        print("\n=== Forward pass on CPU ===")
        model.train()
        logits = model(x)
        print(f"  Logits shape: {logits.shape}")
        check_tensor("Logits", logits)

        loss = torch.nn.functional.cross_entropy(
            logits.squeeze(-1).squeeze(-1), y.squeeze()
        )
        print(f"  Loss: {loss.item():.4f}")

        if torch.isnan(loss):
            print("  WARNING: Loss is NaN on CPU!")
        else:
            print("  Loss is valid on CPU")


if __name__ == "__main__":
    main()
