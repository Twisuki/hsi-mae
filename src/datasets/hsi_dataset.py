"""HSI dataset loader for .npy and .mat formats."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]


class HSIDataset(Dataset):
    """
    Hyperspectral Image dataset loader.

    Supports loading from .npy or .mat files and provides train/val/test splits
    with optional normalization.

    Expected data format:
        data: np.ndarray of shape [H, W, C] where C is the number of bands
        labels: np.ndarray of shape [H, W] (optional)

    Args:
        data_path: Path to .npy or .mat file.
        labels_path: Optional path to labels file.
        mat_key: Key to extract from .mat file (default: "data").
        mat_label_key: Key to extract labels from .mat file (default: "labels").
        split: One of "train", "val", "test".
        train_ratio: Fraction of data for training (default: 0.8).
        val_ratio: Fraction of data for validation (default: 0.1).
        normalize: Whether to normalize to [0, 1] (default: True).
        to_chw: Convert from [H, W, C] to [C, H, W] for PyTorch (default: True).
        unlabeled: If True, labels are all zeros (for pretraining).
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_path: str | Path | None = None,
        *,
        mat_key: str = "data",
        mat_label_key: str = "labels",
        split: Split = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        normalize: bool = True,
        to_chw: bool = True,
        unlabeled: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.mat_key = mat_key
        self.mat_label_key = mat_label_key
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
        self.to_chw = to_chw
        self.unlabeled = unlabeled

        self.data: np.ndarray = self._load_data()
        self.labels: np.ndarray | None = self._load_labels()

        if self.normalize:
            self.data = self._normalize(self.data)

        self.indices = self._compute_split_indices()
        self._validate_shapes()

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _load_data(self) -> np.ndarray:
        """Load HSI data from file."""
        suffix = self.data_path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(self.data_path)
        elif suffix == ".mat":
            from scipy.io import loadmat

            arr = loadmat(self.data_path)[self.mat_key]
        else:
            msg = f"Unsupported file format: {suffix!r} (expected .npy or .mat)"
            raise ValueError(msg)

        return arr.astype(np.float32)

    def _load_labels(self) -> np.ndarray | None:
        """Load label map from file."""
        if self.unlabeled:
            return None

        if self.labels_path is not None:
            suffix = self.labels_path.suffix.lower()
            if suffix == ".npy":
                labels = np.load(self.labels_path)
            elif suffix == ".mat":
                from scipy.io import loadmat

                labels = loadmat(self.labels_path)[self.mat_label_key]
            else:
                msg = f"Unsupported label format: {suffix!r}"
                raise ValueError(msg)
            return labels.astype(np.int64)

        # Labels embedded in .mat
        try:
            from scipy.io import loadmat

            labels = loadmat(self.data_path)[self.mat_label_key]
            return labels.astype(np.int64)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1] per-band."""
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        mins = flat.min(axis=0, keepdims=True)
        maxs = flat.max(axis=0, keepdims=True)
        # Avoid division by zero
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normalized = (flat - mins) / ranges
        return normalized.reshape(shape).astype(np.float32)

    def _compute_split_indices(self) -> np.ndarray:
        """Compute pixel indices for the current split."""
        h, w = self.data.shape[:2]
        indices = np.arange(h * w, dtype=np.int64)

        # Build mask of valid (labeled) pixels
        if self.labels is not None and not self.unlabeled:
            valid = self.labels.ravel() >= 0
            indices = indices[valid]

        np.random.seed(42)  # Deterministic split
        np.random.shuffle(indices)

        n = len(indices)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        if self.split == "train":
            return indices[:n_train]
        elif self.split == "val":
            return indices[n_train : n_train + n_val]
        else:  # "test"
            return indices[n_train + n_val :]

    def _validate_shapes(self) -> None:
        """Ensure data and labels have compatible shapes."""
        dh, dw = self.data.shape[:2]
        if self.labels is not None:
            lh, lw = self.labels.shape
            if lh != dh or lw != dw:
                raise ValueError(
                    f"Data shape {dh}x{dw} does not match labels shape {lh}x{lw}"
                )

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat_idx = self.indices[idx]
        h, w = self.data.shape[:2]
        row, col = flat_idx // w, flat_idx % w

        # Extract pixel spectrum
        spectrum = self.data[row, col]  # shape: [C,]

        if self.to_chw:
            spectrum = spectrum[np.newaxis, ...]  # [1, C] (single-pixel "image")

        x = torch.from_numpy(spectrum.copy())

        # Labels: 0 for unlabeled mode, actual label otherwise
        if self.labels is None or self.unlabeled:
            y = torch.zeros(1, dtype=torch.long)
        else:
            y = torch.tensor(self.labels[row, col], dtype=torch.long)

        return x, y

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Infer number of classes from label map."""
        if self.labels is None:
            return 0
        return int(self.labels.max()) + 1

    @property
    def num_bands(self) -> int:
        """Number of spectral bands."""
        return self.data.shape[2]

    def get_full_data(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return full (unmasked) data and labels as numpy arrays."""
        return self.data, self.labels

    def get_spectral_signature(self, row: int, col: int) -> np.ndarray:
        """Get the spectral signature at a specific pixel."""
        return self.data[row, col].copy()
