#!/usr/bin/env python
"""
Generate synthetic HSI data for quick testing / development.

Creates:
    data/synthetic_hsi.npy        — [H, W, C] float32
    data/synthetic_hsi_gt.npy     — [H, W] int64  labels

Usage:
    python scripts/generate_synthetic_hsi.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_spectral_signature(
    base_index: int,
    num_bands: int = 200,
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate a smooth spectral signature using sine waves + noise."""
    rng = np.random.default_rng(seed + base_index)
    t = np.linspace(0, 4 * np.pi, num_bands)
    freq1 = rng.uniform(1, 3)
    freq2 = rng.uniform(3, 6)
    amp1 = rng.uniform(0.4, 0.8)
    amp2 = rng.uniform(0.1, 0.3)
    phase = rng.uniform(0, 2 * np.pi)
    sig = amp1 * np.sin(freq1 * t + phase) + amp2 * np.cos(freq2 * t)
    sig += rng.normal(0, noise_std, size=num_bands)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    return sig.astype(np.float32)


def generate_hsi(
    height: int = 64,
    width: int = 64,
    num_bands: int = 200,
    num_classes: int = 8,
    structure_size: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a structured synthetic HSI cube with distinct material regions.

    Args:
        height: Image height.
        width: Image height.
        num_bands: Number of spectral bands.
        num_classes: Number of distinct material classes.
        structure_size: Size of each material patch.
        seed: Random seed.

    Returns:
        hsi: [H, W, C] data cube (float32, normalized to [0,1]).
        gt:  [H, W] label map (int64).
    """
    rng = np.random.default_rng(seed)

    # Create label map: grid of patches
    gt = np.zeros((height, width), dtype=np.int64)
    h, w = 0, 0
    class_id = 0
    while h < height and class_id < num_classes:
        patch_h = min(structure_size + rng.integers(-2, 3), height - h)
        while w < width and class_id < num_classes:
            patch_w = min(structure_size + rng.integers(-2, 3), width - w)
            gt[h : h + patch_h, w : w + patch_w] = class_id
            class_id += 1
            w += patch_w
        h += patch_h
        w = 0

    # Assign spectral signature per class
    class_spectra = {
        c: generate_spectral_signature(c, num_bands, seed=seed)
        for c in range(num_classes)
    }

    # Build HSI cube
    hsi = np.zeros((height, width, num_bands), dtype=np.float32)
    for c in range(num_classes):
        mask = gt == c
        sig = class_spectra[c]
        hsi[mask] = sig

    # Add spatial texture variation
    noise = rng.normal(0, 0.03, size=hsi.shape).astype(np.float32)
    hsi = np.clip(hsi + noise, 0, 1)

    # Per-band normalization
    for b in range(num_bands):
        band = hsi[:, :, b]
        hsi[:, :, b] = (band - band.min()) / (band.max() - band.min() + 1e-8)

    return hsi.astype(np.float32), gt.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic HSI data")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--bands", type=int, default=200)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--structure-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    print(
        f"Generating HSI {args.height}x{args.width}x{args.bands}, "
        f"{args.num_classes} classes..."
    )

    hsi, gt = generate_hsi(
        height=args.height,
        width=args.width,
        num_bands=args.bands,
        num_classes=args.num_classes,
        structure_size=args.structure_size,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "synthetic_hsi.npy"
    gt_path = out_dir / "synthetic_hsi_gt.npy"

    np.save(data_path, hsi)
    np.save(gt_path, gt)

    print(f"Data saved:   {data_path}  shape={hsi.shape}")
    print(f"Labels saved: {gt_path}    shape={gt.shape}")
    print(f"Unique labels: {np.unique(gt)}")
    print("Done.")


if __name__ == "__main__":
    main()
