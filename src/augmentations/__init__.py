"""Augmentations package."""

from src.augmentations.hsi_mask import (
    JointMask,
    MaskStrategy,
    RandomMask,
    SpatialMask,
    SpectralMask,
    apply_joint_mask,
    apply_random_mask,
    apply_spatial_mask,
    apply_spectral_mask,
)

__all__ = [
    "MaskStrategy",
    "SpectralMask",
    "SpatialMask",
    "JointMask",
    "RandomMask",
    "apply_spectral_mask",
    "apply_spatial_mask",
    "apply_joint_mask",
    "apply_random_mask",
]
