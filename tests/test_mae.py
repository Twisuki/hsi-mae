"""Tests for HSIMAE."""

import pytest
import torch

from src.models.mae import HSIMAE


class TestHSIMAE:
    """Tests for HSIMAE model."""

    def test_forward_returns_three_tensors(self):
        model = HSIMAE(bands=200, encoder_dim=128, mask_ratio=0.75)
        x = torch.randn(2, 200, 8, 8)
        recon, x_masked, mask = model(x)

        assert recon.shape == x.shape
        assert x_masked.shape == x.shape
        assert mask.shape == x.shape
        assert mask.dtype == torch.bool

    def test_masked_values_zero(self):
        model = HSIMAE(bands=50, mask_ratio=0.75)
        x = torch.randn(2, 50, 8, 8)
        _, x_masked, mask = model(x)
        # Masked positions should be zero
        masked_vals = x_masked[~mask]
        assert torch.allclose(masked_vals, torch.zeros_like(masked_vals), atol=1e-6)

    def test_encode_no_mask_passes_through(self):
        model = HSIMAE(bands=50, encoder_dim=64)
        x = torch.randn(2, 50, 8, 8)
        z = model.encode_no_mask(x)
        assert z.shape == (2, 64, 8, 8)

    def test_decode_features(self):
        model = HSIMAE(bands=50, encoder_dim=64)
        z = torch.randn(2, 64, 8, 8)
        recon = model.decode_features(z)
        assert recon.shape == (2, 50, 8, 8)

    def test_encoder_attribute(self):
        model = HSIMAE(bands=50, encoder_dim=64)
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "mask_generator")

    def test_backward(self):
        model = HSIMAE(bands=50, encoder_dim=64)
        x = torch.randn(2, 50, 8, 8, requires_grad=True)
        recon, _, _ = model(x)
        loss = recon.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_mask_ratios(self):
        for ratio in [0.3, 0.5, 0.75, 0.9]:
            model = HSIMAE(bands=50, mask_ratio=ratio)
            x = torch.randn(2, 50, 8, 8)
            _, _, mask = model(x)
            kept_ratio = mask.float().mean().item()
            # Approximately match the mask ratio
            assert 0.1 < kept_ratio < 0.9

    def test_config_attributes(self):
        model = HSIMAE(bands=150, encoder_dim=128, mask_ratio=0.6)
        assert model.bands == 150
        assert model.encoder_dim == 128
        assert model.mask_ratio == 0.6
