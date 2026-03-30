"""Tests for SAM loss functions."""

import pytest
import torch

from src.losses.sam import (
    MSELS,
    MSESAMLoss,
    SAMLoss,
    mse_loss,
    mse_sam_loss,
    sam_loss,
)


class TestSAMLoss:
    """Tests for sam_loss function."""

    def test_identical_input_zero_loss(self):
        x = torch.randn(4, 200, 8, 8)
        loss = sam_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_spectra_max_loss(self):
        # Two orthogonal spectra should give SAM ≈ 0.5 (90° = π/2, /π ≈ 0.5)
        spectrum = torch.tensor([[1.0, 0.0, 0.0]])
        ortho = torch.tensor([[0.0, 1.0, 0.0]])
        loss = sam_loss(spectrum, ortho)
        assert 0.4 < loss.item() < 0.6

    def test_same_direction_min_loss(self):
        spectrum = torch.randn(1, 10)
        loss = sam_loss(spectrum, spectrum * 2.0)  # scaling should not affect angle
        assert loss.item() < 1e-6

    def test_shape_preserved(self):
        x = torch.randn(2, 100, 5, 5)
        loss = sam_loss(x, x)
        assert loss.ndim == 0  # scalar

    def test_backward(self):
        x = torch.randn(2, 50, 4, 4, requires_grad=True)
        target = torch.randn(2, 50, 4, 4)
        loss = sam_loss(x, target)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMSELoss:
    def test_identical_zero_loss(self):
        x = torch.randn(2, 200, 3, 3)
        assert mse_loss(x, x).item() == pytest.approx(0.0)

    def test_backward(self):
        x = torch.randn(2, 200, 3, 3, requires_grad=True)
        target = torch.randn(2, 200, 3, 3)
        loss = mse_loss(x, target)
        loss.backward()
        assert x.grad is not None


class TestMSESAMLoss:
    def test_combined_loss_nonzero(self):
        x = torch.randn(2, 200, 4, 4)
        target = torch.randn(2, 200, 4, 4)
        loss = mse_sam_loss(x, target, lambda_sam=0.1)
        assert loss.item() >= 0.0

    def test_lambda_zero_equals_mse(self):
        x = torch.randn(2, 200, 4, 4)
        target = torch.randn(2, 200, 4, 4)
        mse_only = mse_loss(x, target)
        combined = mse_sam_loss(x, target, lambda_sam=0.0)
        assert combined.item() == pytest.approx(mse_only.item(), rel=1e-5)


class TestLossModules:
    def test_sam_module(self):
        module = SAMLoss()
        x = torch.randn(2, 100, 3, 3)
        loss = module(x, x)
        assert loss.item() < 1e-5

    def test_mse_module(self):
        module = MSELS()
        x = torch.randn(2, 100, 3, 3)
        assert module(x, x).item() == pytest.approx(0.0)

    def test_combined_module(self):
        module = MSESAMLoss(lambda_sam=0.5)
        x = torch.randn(2, 100, 3, 3)
        target = torch.randn(2, 100, 3, 3)
        loss = module(x, target)
        assert loss.item() >= 0.0
