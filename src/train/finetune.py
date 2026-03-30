"""Fine-tuning engine for HSI classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.classifier import HSIFineTuner
from src.models.encoder import HSIEncoder
from src.utils.logger import get_logger

# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall Accuracy (OA)."""
    return (y_true == y_pred).mean()


def average_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """Average Accuracy (AA) — mean recall across classes."""
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append((y_true[mask] == y_pred[mask]).mean())
    return np.mean(recalls) if recalls else 0.0


def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kappa coefficient."""
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    sum_observed = cm.diagonal().sum()
    sum_expected = cm.sum(axis=1).dot(cm.sum(axis=0)) / n
    return (sum_observed - sum_expected) / (n - sum_expected)


def f1_scores(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Per-class and macro F1 scores."""
    f1_per_class: list[float] = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    return {
        "f1_macro": float(np.mean(f1_per_class)),
        **{f"f1_class_{i}": f for i, f in enumerate(f1_per_class)},
    }


# -------------------------------------------------------------------------
# FinetuneEngine
# -------------------------------------------------------------------------

class FinetuneEngine:
    """
    Fine-tuning engine for HSI classification.

    Supports two modes:
        - "linear_probe": Freeze encoder, train classifier head only.
        - "full": Fine-tune encoder + classifier jointly.

    Args:
        encoder: Pretrained HSIEncoder instance.
        num_classes: Number of land-cover classes.
        mode: Fine-tune mode — "linear_probe" or "full".
        lr: Learning rate (default: 1e-3).
        weight_decay: Weight decay (default: 1e-4).
        device: Device (default: "cuda").
        save_dir: Directory to save checkpoints.
        log_interval: Steps between log prints.
        seed: Random seed.
    """

    def __init__(
        self,
        encoder: HSIEncoder,
        num_classes: int,
        mode: str = "full",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        log_interval: int = 10,
        seed: int = 42,
    ) -> None:
        if mode not in ("linear_probe", "full"):
            msg = f"mode must be 'linear_probe' or 'full', got {mode!r}"
            raise ValueError(msg)

        self.mode = mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.logger = get_logger()
        self.global_step = 0
        self.epoch = 0

        freeze_encoder = (mode == "linear_probe")
        self.model = HSIFineTuner(
            encoder=encoder,
            num_classes=num_classes,
            classifier_variant="mlp",
            freeze_encoder=freeze_encoder,
        ).to(self.device)

        params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.logger.info(
            f"FinetuneEngine | mode={mode} | device={self.device} | "
            f"encoder_dim={encoder.get_output_dim()} | num_classes={num_classes}"
        )

    # -------------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------------

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(x)  # [B, num_classes, 1, 1]

        # Squeeze spatial dims (pixel-level classification)
        logits = logits.squeeze(-1).squeeze(-1)
        y = y.squeeze()

        loss = self.criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()

        return {"loss": loss.item(), "acc": acc}

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
        Run fine-tuning.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Number of epochs.
        """
        best_metrics: dict[str, float] = {"oa": 0.0}

        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            epoch_losses: list[float] = []
            epoch_accs: list[float] = []

            for batch_idx, (x, y) in enumerate(pbar):
                metrics = self._train_step(x, y)
                epoch_losses.append(metrics["loss"])
                epoch_accs.append(metrics["acc"])
                self.global_step += 1

                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        acc=f"{metrics['acc']:.2%}",
                    )

            train_loss = np.mean(epoch_losses)
            train_acc = np.mean(epoch_accs)

            # ---- Validate ----
            val_metrics: dict[str, float] | None = None
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader)
                if val_metrics["oa"] > best_metrics["oa"]:
                    best_metrics = val_metrics
                    self._save_checkpoint("best_classifier.pt")

            # ---- Log ----
            lr_now = self.optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={train_loss:.4f} | acc={train_acc:.2%} | lr={lr_now:.2e}"
            )
            if val_metrics:
                msg += (
                    f" | val_oa={val_metrics['oa']:.2%} | "
                    f"val_aa={val_metrics['aa']:.2%} | "
                    f"val_kappa={val_metrics['kappa']:.4f}"
                )
            self.logger.info(msg)

            self._save_checkpoint("last_classifier.pt")

        self.logger.info(
            f"Best OA={best_metrics['oa']:.2%} | "
            f"AA={best_metrics['aa']:.2%} | "
            f"Kappa={best_metrics['kappa']:.4f}"
        )

    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        losses: list[float] = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device).squeeze()

                logits = self.model(x).squeeze(-1).squeeze(-1)
                loss = self.criterion(logits, y)
                losses.append(loss.item())

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": float(np.mean(losses)),
            "oa": overall_accuracy(all_labels, all_preds),
            "aa": average_accuracy(all_labels, all_preds, self.num_classes),
            "kappa": kappa_coefficient(all_labels, all_preds),
            **f1_scores(all_labels, all_preds, self.num_classes),
        }
        return metrics

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.epoch,
            },
            path,
        )
        self.logger.debug(f"Checkpoint saved: {path}")

    def load_classifier(self, path: str | Path) -> None:
        """Load classifier weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(f"Classifier loaded from {path}")

    @property
    def fine_tuner(self) -> HSIFineTuner:
        """Expose the fine-tuning model."""
        return self.model
