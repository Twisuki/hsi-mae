"""Mask strategies for HSI-MAE."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskStrategy(nn.Module):
    """
    Base class for HSI masking strategies.

    All strategies return the masked tensor and a boolean mask indicating
    which elements were kept (True = kept, False = masked).
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to input.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, H, W, C].

        Returns:
            x_masked: Masked tensor, same shape as input.
            mask: Boolean mask, same shape as input (True = kept).
        """
        raise NotImplementedError


class SpectralMask(MaskStrategy):
    """
    Randomly mask entire spectral bands.

    Each band is independently kept with probability (1 - spectral_ratio).
    """

    def __init__(self, spectral_ratio: float = 0.5) -> None:
        super().__init__()
        self.spectral_ratio = spectral_ratio

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        mask = torch.rand(B, C, 1, 1, device=x.device) > self.spectral_ratio
        x_masked = x.clone()
        x_masked = x_masked * mask.to(x.dtype)
        return x_masked, mask


class SpatialMask(MaskStrategy):
    """
    Randomly mask spatial patches.

    Each spatial location is independently kept with probability (1 - spatial_ratio).
    """

    def __init__(self, spatial_ratio: float = 0.5) -> None:
        super().__init__()
        self.spatial_ratio = spatial_ratio

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        mask = torch.rand(B, 1, H, W, device=x.device) > self.spatial_ratio
        x_masked = x.clone()
        x_masked = x_masked * mask.to(x.dtype)
        return x_masked, mask


class JointMask(MaskStrategy):
    """
    3D joint masking: spectral mask AND spatial mask are applied together.

    Elements are kept only if they survive both the spectral and spatial masks,
    resulting in a sparser representation.
    """

    def __init__(self, spectral_ratio: float = 0.5, spatial_ratio: float = 0.5) -> None:
        super().__init__()
        self.spectral_ratio = spectral_ratio
        self.spatial_ratio = spatial_ratio

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        # Spectral mask: [B, C, 1, 1]
        s_mask = torch.rand(B, C, 1, 1, device=x.device) > self.spectral_ratio
        # Spatial mask: [B, 1, H, W]
        p_mask = torch.rand(B, 1, H, W, device=x.device) > self.spatial_ratio
        # Joint: both must keep
        mask = s_mask & p_mask
        x_masked = x.clone()
        x_masked = x_masked * mask.to(x.dtype)
        return x_masked, mask


class RandomMask(MaskStrategy):
    """
    Random element-wise masking with a single ratio.

    Each element is independently kept with probability (1 - mask_ratio).
    Equivalent to applying a Bernoulli mask across the entire tensor.
    """

    def __init__(self, mask_ratio: float = 0.75) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.rand_like(x) > self.mask_ratio
        x_masked = x.clone()
        x_masked = x_masked * mask.to(x.dtype)
        return x_masked, mask


def apply_spectral_mask(
    x: torch.Tensor, ratio: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for spectral masking."""
    return SpectralMask(spectral_ratio=ratio)(x)


def apply_spatial_mask(
    x: torch.Tensor, ratio: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for spatial masking."""
    return SpatialMask(spatial_ratio=ratio)(x)


def apply_joint_mask(
    x: torch.Tensor,
    spectral_ratio: float = 0.5,
    spatial_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for joint masking."""
    return JointMask(spectral_ratio=spectral_ratio, spatial_ratio=spatial_ratio)(x)


def apply_random_mask(
    x: torch.Tensor, mask_ratio: float = 0.75
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for random masking."""
    return RandomMask(mask_ratio=mask_ratio)(x)
