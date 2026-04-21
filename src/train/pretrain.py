"""Pre-training engine for HSI-MAE."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.sam import mse_sam_loss
from src.models.mae import HSIMAE
from src.utils.logger import get_logger


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
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
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

        for epoch in range(epochs):
            self.epoch = epoch
            self._adjust_lr(epoch, epochs)

            # ---- Train ----
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            epoch_losses: list[float] = []

            for batch_idx, (x, _) in enumerate(pbar):
                metrics = self._train_step(x)
                epoch_losses.append(metrics["loss"])
                self.global_step += 1

                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        grad=f"{metrics['grad_norm']:.2f}",
                        lr=f"{metrics['lr']:.2e}",
                    )

            train_loss = sum(epoch_losses) / len(epoch_losses)

            # ---- Validate ----
            val_loss: float | None = None
            if val_loader is not None and epoch % self.val_interval == 0:
                val_loss = self._validate(val_loader)
                if not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best_encoder.pt")

            # Always save best on first epoch as fallback
            if epoch == 0:
                self._save_checkpoint("best_encoder.pt")

            # ---- Log epoch ----
            lr_now = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.4f}"
            msg += f" | lr={lr_now:.2e}"
            self.logger.info(msg)

            # Save latest
            self._save_checkpoint("last_encoder.pt")

        self.logger.info("Pre-training complete.")

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        losses: list[float] = []

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(self.device)
                recon, _, _ = self.model(x)
                loss = mse_sam_loss(recon, x, lambda_sam=0.1)
                losses.append(loss.item())

        return sum(losses) / len(losses)

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
