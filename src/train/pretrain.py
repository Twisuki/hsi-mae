"""Pre-training engine for HSI-MAE."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.sam import mse_sam_loss
from src.models.mae import HSIMAE
from src.utils.logger import get_logger
from src.utils.visualization import plot_reconstruction_progress, plot_training_curve


class PretrainEngine:
    """
    Pre-training engine for HSI-MAE.

    Handles:
        - Model initialization
        - Learning rate scheduling (linear warmup + cosine decay)
        - Gradient clipping
        - Checkpoint saving (encoder only)
        - Logging

    Args:
        bands: Number of spectral bands.
        encoder_dim: Encoder embedding dimension.
        encoder_layers: Number of encoder spatial layers.
        decoder_dim: Decoder hidden dimension.
        mask_ratio: Fraction of elements to mask (default: 0.75).
        lr: Peak learning rate (default: 1e-3).
        weight_decay: Weight decay (default: 1e-4).
        warmup_epochs: Warmup epochs (default: 10).
        device: Device to train on (default: "cuda").
        save_dir: Directory to save checkpoints (default: "checkpoints").
        log_interval: Steps between log prints (default: 10).
        val_interval: Epochs between validation runs (default: 1).
        seed: Random seed.
    """

    def __init__(
        self,
        bands: int,
        encoder_dim: int = 256,
        encoder_layers: int = 4,
        decoder_dim: int = 256,
        mask_ratio: float = 0.75,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        log_interval: int = 10,
        val_interval: int = 1,
        seed: int = 42,
        visualize: bool = True,
    ) -> None:
        self.bands = bands
        self.encoder_dim = encoder_dim
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.seed = seed
        self.visualize = visualize

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Build model
        self.model = HSIMAE(
            bands=bands,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            decoder_dim=decoder_dim,
            mask_ratio=mask_ratio,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.logger = get_logger()
        self.global_step = 0
        self.epoch = 0

        # Visualization history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_metrics: list[float] = []
        self.val_metrics: list[float] = []
        self.recon_samples: list[dict[str, Any]] = []  # Store original/recon for visualization

    # -------------------------------------------------------------------------
    # Learning rate schedule
    # -------------------------------------------------------------------------

    def _get_lr(self, epoch: int, total_epochs: int) -> float:
        """Linear warmup + cosine annealing."""
        warmup_steps = self.warmup_epochs
        if epoch < warmup_steps:
            return self.lr * (epoch + 1) / warmup_steps
        progress = (epoch - warmup_steps) / max(1, total_epochs - warmup_steps)
        return self.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _adjust_lr(self, epoch: int, total_epochs: int) -> None:
        lr = self._get_lr(epoch, total_epochs)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # -------------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------------

    def _train_step(self, batch: torch.Tensor) -> dict[str, float]:
        self.model.train()
        x = batch.to(self.device)  # [B, bands, 1, 1] from pixel dataset

        self.optimizer.zero_grad()
        recon, x_masked, mask = self.model(x)

        loss = mse_sam_loss(recon, x, lambda_sam=0.1)

        # Check for nan loss
        if torch.isnan(loss):
            self.logger.warning("Loss is NaN, skipping this step")
            return {
                "loss": float("nan"),
                "grad_norm": 0.0,
                "lr": self.optimizer.param_groups[0]["lr"],
                "recon": recon.detach().cpu(),
                "original": x.detach().cpu(),
            }

        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Check for exploding gradients
        if grad_norm > 100:
            self.logger.warning(f"Exploding gradient: grad_norm={grad_norm:.2f}")

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "recon": recon.detach().cpu(),
            "original": x.detach().cpu(),
        }

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
    ) -> None:
        """
        Run pre-training.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Number of training epochs.
        """
        self.logger.info(
            f"Start pre-training: {epochs} epochs, device={self.device}, "
            f"lr={self.lr}, mask_ratio={self.mask_ratio}"
        )

        best_val_loss = float("inf")

        # Clear history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.recon_samples = []

        for epoch in range(epochs):
            self.epoch = epoch
            self._adjust_lr(epoch, epochs)

            # ---- Train ----
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            epoch_losses: list[float] = []
            epoch_recon: torch.Tensor | None = None
            epoch_original: torch.Tensor | None = None

            for batch_idx, (x, _) in enumerate(pbar):
                metrics = self._train_step(x)
                epoch_losses.append(metrics["loss"])
                self.global_step += 1

                # Store first sample of first batch for visualization
                if batch_idx == 0 and self.visualize:
                    epoch_original = metrics["original"][0:1]
                    epoch_recon = metrics["recon"][0:1]

                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        grad=f"{metrics['grad_norm']:.2f}",
                        lr=f"{metrics['lr']:.2e}",
                    )

            train_loss = sum(epoch_losses) / len(epoch_losses)
            self.train_losses.append(train_loss)

            # Calculate a simple "accuracy" metric (negative normalized MSE)
            train_metric = 1.0 / (1.0 + train_loss)
            self.train_metrics.append(train_metric)

            # ---- Validate ----
            val_loss: float | None = None
            val_metric: float | None = None
            if val_loader is not None and epoch % self.val_interval == 0:
                val_loss, val_metric = self._validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metric if val_metric is not None else 0.0)

                if not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best_encoder.pt")

            # Always save best on first epoch as fallback
            if epoch == 0:
                self._save_checkpoint("best_encoder.pt")

            # Store reconstruction sample for visualization
            if self.visualize and epoch_original is not None and epoch_recon is not None:
                self.recon_samples.append({
                    "original": epoch_original[0].numpy(),
                    "reconstruction": epoch_recon[0].numpy(),
                    "epoch": epoch + 1,
                })

            # ---- Log epoch ----
            lr_now = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.4f}"
            msg += f" | lr={lr_now:.2e}"
            self.logger.info(msg)

            # Save latest
            self._save_checkpoint("last_encoder.pt")

        # ---- Final visualization ----
        if self.visualize:
            self._save_visualizations()

        self.logger.info("Pre-training complete.")

    def _validate(self, val_loader: DataLoader) -> tuple[float, float | None]:
        self.model.eval()
        losses: list[float] = []

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(self.device)
                recon, _, _ = self.model(x)
                loss = mse_sam_loss(recon, x, lambda_sam=0.1)
                losses.append(loss.item())

        val_loss = sum(losses) / len(losses)
        val_metric = 1.0 / (1.0 + val_loss)  # Simple normalized metric
        return val_loss, val_metric

    def _save_visualizations(self) -> None:
        """Save all visualization plots after training."""
        # 1. Training curve (loss + metric)
        if self.train_losses:
            plot_training_curve(
                train_losses=self.train_losses,
                val_losses=self.val_losses if self.val_losses else None,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics if self.val_metrics else None,
                metric_name="Score (1/(1+loss))",
                title="HSI-MAE Pre-training",
                save_path=self.save_dir / "training_curve.png",
            )

        # 2. Reconstruction progress
        if self.recon_samples:
            originals = [s["original"] for s in self.recon_samples]
            reconstructions = [s["reconstruction"] for s in self.recon_samples]
            epoch_labels = [f"Epoch {s['epoch']}" for s in self.recon_samples]
            plot_reconstruction_progress(
                originals=originals,
                reconstructions=reconstructions,
                epoch_labels=epoch_labels,
                num_bands=self.bands,
                title="Reconstruction Progress",
                save_path=self.save_dir / "reconstruction_progress.png",
            )

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "encoder_state": self.model.encoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": {
                    "bands": self.bands,
                    "encoder_dim": self.encoder_dim,
                    "mask_ratio": self.mask_ratio,
                },
            },
            path,
        )
        self.logger.debug(f"Checkpoint saved: {path}")

    def load_encoder(self, path: str | Path) -> None:
        """Load encoder weights from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.encoder.load_state_dict(ckpt["encoder_state"])
        self.logger.info(f"Encoder loaded from {path}")

    @property
    def encoder(self) -> nn.Module:
        """Expose encoder for downstream use."""
        return self.model.encoder
