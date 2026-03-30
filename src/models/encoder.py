"""HSI Encoder: spectral and spatial feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn


class HSIEncoder(nn.Module):
    """
    Hyperspectral Image Encoder with Conv2d spectral + spatial branches.

    Architecture:
        Input [B, bands, H, W]
        → Spectral embedding: Conv2d(bands → dim, kernel=1)
        → Spatial encoder: N × (Conv2d(dim→dim, k=3, p=1) + BN + ReLU)
        → Output [B, dim, H, W]

    Args:
        bands: Number of input spectral bands.
        dim: Embedding dimension (default: 256).
        num_layers: Number of spatial encoder blocks (default: 4).
        norm: Normalization layer type ("bn", "gn", "none"). Default: "bn".
    """

    def __init__(
        self,
        bands: int,
        dim: int = 256,
        num_layers: int = 4,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.bands = bands
        self.dim = dim
        self.num_layers = num_layers

        # Spectral embedding: [B, bands, H, W] → [B, dim, H, W]
        self.spectral_proj = nn.Conv2d(bands, dim, kernel_size=1)

        # Spatial encoder blocks
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_c = dim if i > 0 else dim
            block = nn.Sequential(
                nn.Conv2d(in_c, dim, kernel_size=3, padding=1, bias=False),
                self._make_norm(dim, norm),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                self._make_norm(dim, norm),
            )
            layers.append(block)

        self.spatial_encoder = nn.ModuleList(layers)

    @staticmethod
    def _make_norm(num_features: int, norm: str) -> nn.Module:
        if norm == "bn":
            return nn.BatchNorm2d(num_features)
        if norm == "gn":
            return nn.GroupNorm(num_groups=8, num_channels=num_features)
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input HSI to latent features.

        Args:
            x: Input tensor [B, bands, H, W].

        Returns:
            Encoded features [B, dim, H, W].
        """
        x = self.spectral_proj(x)  # [B, dim, H, W]

        for block in self.spatial_encoder:
            x = x + block(x)  # Residual-style update

        return x

    def get_output_dim(self) -> int:
        """Return the encoder output feature dimension."""
        return self.dim

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
