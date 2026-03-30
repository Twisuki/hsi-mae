"""Training-time augmentation pipeline for HSI data."""

from __future__ import annotations

import numpy as np
import torch


class TrainAugmentation:
    """
    Training augmentations for HSI pixel data.

    Applies random spectral/spatial transforms to the input tensor.
    This runs inside the training loop (on GPU tensors when using
    DataLoader pre-fetching, or CPU tensors otherwise).

    Augmentations applied (in order):
        1. Random spectral noise (Gaussian, small amplitude)
        2. Random spectral shift (shift band values slightly)
        3. Random spatial dropout (drop entire bands)
        4. Random channel shuffle (swap two bands)

    Args:
        noise_std: Std dev of Gaussian noise added to spectrum (default: 0.01).
        shift_range: Max absolute shift per band (default: 0.05).
        dropout_prob: Probability of dropping each band (default: 0.05).
        shuffle_prob: Probability of swapping two random bands (default: 0.05).
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        shift_range: float = 0.05,
        dropout_prob: float = 0.05,
        shuffle_prob: float = 0.05,
    ) -> None:
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.dropout_prob = dropout_prob
        self.shuffle_prob = shuffle_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations.

        Args:
            x: Tensor of shape [B, C] or [B, C, 1, 1].

        Returns:
            Augmented tensor, same shape.
        """
        if x.ndim == 4:
            x = x.squeeze(-1).squeeze(-1)  # [B, C]

        x = x.clone()

        # 1. Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 2. Random spectral shift
        if self.shift_range > 0:
            shift = torch.empty_like(x).uniform_(
                -self.shift_range, self.shift_range
            )
            x = x + shift

        # 3. Random band dropout
        if self.dropout_prob > 0:
            mask = torch.rand_like(x) > self.dropout_prob
            x = x * mask.to(x.dtype)

        # 4. Random band shuffle
        if self.shuffle_prob > 0 and x.shape[0] > 1:
            B = x.shape[0]
            for b in range(B):
                if torch.rand(1).item() < self.shuffle_prob:
                    i, j = torch.randint(0, x.shape[1], (2,)).tolist()
                    x[b, i], x[b, j] = x[b, j], x[b, i]

        # Restore shape
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x


class ValAugmentation:
    """
    Validation augmentations — identity (no-op).

    Use this for val/test DataLoader to ensure deterministic transforms.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Compose:
    """Compose multiple augmentation callables."""

    def __init__(self, *transforms: callable) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    """Convert numpy array to torch tensor (float32)."""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32))


class Normalize:
    """Min-max normalize tensor to [0, 1] per-sample."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Per-sample normalize over band dimension
        mins = x.min(dim=1, keepdim=True)[0]
        maxs = x.max(dim=1, keepdim=True)[0]
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (x - mins) / ranges
