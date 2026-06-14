"""Spectral Angle Mapper (SAM) loss and variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spectral Angle Mapper loss.

    Measures the spectral angle between predicted and target spectra.
    Lower is better. Range: [0, 1].

    .. math::
        \\text{SAM}(x, \\hat{x}) = \\frac{1}{\\pi} \\arccos
        \\left( \\frac{\\langle x, \\hat{x} \\rangle}
        {\\|x\\| \\|\\hat{x}\\|} \\right)

    The arccos / π normalization maps [0, π] → [0, 1].

    Args:
        pred: Predicted tensor, shape [B, C, H, W] where C is the number of bands.
        target: Target tensor, same shape as pred.

    Returns:
        Scalar SAM loss (mean over all spatial pixels and batch).
    """
    # Flatten spatial dims: [B, C, H, W] → [B*H*W, C]
    flat_pred = pred.reshape(-1, pred.shape[1])
    flat_target = target.reshape(-1, target.shape[1])

    # L2 normalize along band axis
    norm_pred = F.normalize(flat_pred, dim=1)
    norm_target = F.normalize(flat_target, dim=1)

    # Cosine similarity (inner product of normalized vectors)
    cos_sim = (norm_pred * norm_target).sum(dim=1)

    # Clamp to [-1, 1] for numerical stability
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    # Arccos → radians; divide by π to map [0, π] → [0, 1]
    sam = torch.acos(cos_sim) / torch.pi

    return sam.mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss (helper, avoids extra import)."""
    return F.mse_loss(pred, target)


def mse_sam_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_sam: float = 0.1,
) -> torch.Tensor:
    """
    Combined MSE + SAM loss for HSI-MAE training.

    .. math::
        L = \\text{MSE}(\\hat{x}, x) + \\lambda \\cdot \\text{SAM}(\\hat{x}, x)

    Args:
        pred: Predicted tensor, shape [B, C, H, W].
        target: Target tensor, shape [B, C, H, W].
        lambda_sam: Weight for SAM term (default: 0.1).

    Returns:
        Scalar combined loss.
    """
    mse = mse_loss(pred, target)
    sam = sam_loss(pred, target)
    return mse + lambda_sam * sam


class MSELS(nn.Module):
    """MSE loss as an nn.Module for use with optimizers."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_loss(pred, target)


class SAMLoss(nn.Module):
    """SAM loss as an nn.Module."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sam_loss(pred, target)


class MSESAMLoss(nn.Module):
    """
    Combined MSE + SAM loss as an nn.Module.

    Args:
        lambda_sam: Weight for SAM term (default: 0.1).
    """

    def __init__(self, lambda_sam: float = 0.1) -> None:
        super().__init__()
        self.lambda_sam = lambda_sam

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_sam_loss(pred, target, self.lambda_sam)
