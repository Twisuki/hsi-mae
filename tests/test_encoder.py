"""Tests for HSIEncoder."""

import pytest
import torch

from src.models.encoder import HSIEncoder


class TestHSIEncoder:
    """Tests for HSIEncoder."""

    def test_output_shape(self):
        encoder = HSIEncoder(bands=200, dim=256, num_layers=4)
        x = torch.randn(4, 200, 8, 8)
        out = encoder(x)
        assert out.shape == (4, 256, 8, 8)

    def test_output_dim_property(self):
        encoder = HSIEncoder(bands=100, dim=128)
        assert encoder.get_output_dim() == 128

    def test_bands_and_dim_attributes(self):
        encoder = HSIEncoder(bands=150, dim=64, num_layers=2)
        assert encoder.bands == 150
        assert encoder.dim == 64
        assert encoder.num_layers == 2

    def test_different_spatial_sizes(self):
        encoder = HSIEncoder(bands=50, dim=64)
        for h, w in [(8, 8), (16, 16), (4, 12)]:
            x = torch.randn(2, 50, h, w)
            out = encoder(x)
            assert out.shape == (2, 64, h, w)

    def test_freeze_unfreeze(self):
        encoder = HSIEncoder(bands=50, dim=64)
        # Initially trainable
        assert all(p.requires_grad for p in encoder.parameters())

        encoder.freeze()
        assert not any(p.requires_grad for p in encoder.parameters())

        encoder.unfreeze()
        assert all(p.requires_grad for p in encoder.parameters())

    def test_backward(self):
        encoder = HSIEncoder(bands=50, dim=64)
        x = torch.randn(2, 50, 8, 8, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert encoder.spectral_proj.weight.grad is not None

    @pytest.mark.parametrize("norm", ["bn", "gn", "none"])
    def test_norm_types(self, norm):
        encoder = HSIEncoder(bands=50, dim=64, norm=norm)
        x = torch.randn(2, 50, 8, 8)
        out = encoder(x)
        assert out.shape == (2, 64, 8, 8)
