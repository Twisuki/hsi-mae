"""HSI-MAE: Masked Autoencoder for Hyperspectral Images."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.augmentations.hsi_mask import RandomMask
from src.models.encoder import HSIEncoder


class HSIMAE(nn.Module):
    """
    HSI Masked Autoencoder.

    Composed of:
        1. Encoder: extracts latent features from masked input
        2. Decoder: reconstructs the original spectrum from latent

    Forward pass applies RandomMask internally before encoding.

    Args:
        bands: Number of input spectral bands.
        encoder_dim: Encoder embedding dimension.
        encoder_layers: Number of spatial encoder layers.
        decoder_dim: Decoder hidden dimension (default: same as encoder_dim).
        mask_ratio: Fraction of elements to mask (default: 0.75).
        norm: Normalization type for encoder ("bn", "gn", "none").
    """

    def __init__(
        self,
        bands: int,
        encoder_dim: int = 256,
        encoder_layers: int = 4,
        decoder_dim: int | None = None,
        mask_ratio: float = 0.75,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.bands = bands
        self.encoder_dim = encoder_dim
        self.mask_ratio = mask_ratio

        decoder_dim = decoder_dim or encoder_dim

        # Encoder
        self.encoder = HSIEncoder(
            bands=bands,
            dim=encoder_dim,
            num_layers=encoder_layers,
            norm=norm,
        )

        # Decoder: latent → band reconstruction
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_dim, decoder_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, bands, kernel_size=1),
        )

        # Mask generator (no parameters)
        self.mask_generator = RandomMask(mask_ratio=mask_ratio)

        # Initialize decoder weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input (no masking)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to spectral reconstruction."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE forward: mask → encode → decode.

        Args:
            x: Input [B, bands, H, W].

        Returns:
            recon: Reconstructed spectrum [B, bands, H, W].
            x_masked: Masked input [B, bands, H, W].
            mask: Boolean mask [B, bands, H, W] (True = kept).
        """
        # Apply masking
        x_masked, mask = self.mask_generator(x)

        # Encode
        z = self.encoder(x_masked)

        # Decode
        recon = self.decoder(z)

        return recon, x_masked, mask

    def encode_no_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without masking (use for inference / classifier)."""
        return self.encoder(x)

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features (expose for downstream tasks)."""
        return self.decoder(z)
