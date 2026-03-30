"""Configuration management for HSI-MAE."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""

    data_path: str = ""
    train_split: float = 0.8
    val_split: float = 0.1
    num_classes: int = 16
    patch_size: int = 8
    normalize: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    bands: int = 200
    encoder_dim: int = 256
    encoder_layers: int = 4
    decoder_dim: int = 256
    use_vit: bool = False
    patch_size: int = 8
    mask_ratio: float = 0.75


@dataclass
class TrainConfig:
    """Pre-training configuration."""

    # Model
    bands: int = 200
    encoder_dim: int = 256
    encoder_layers: int = 4
    decoder_dim: int = 256
    patch_size: int = 8
    mask_ratio: float = 0.75

    # Data
    data_path: str = ""
    train_split: float = 0.8
    val_split: float = 0.1
    patch_size_data: int = 8
    normalize: bool = True

    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    lambda_sam: float = 0.1

    # System
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    save_dir: str = "checkpoints"
    log_interval: int = 10
    val_interval: int = 1

    # Fine-tune specific
    finetune_mode: str = "full"  # "linear_probe" | "full"
    encoder_path: str = ""


@dataclass
class ConfigManager:
    """Manages configuration loading and saving."""

    config: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> ConfigManager:
        """Create config from command-line arguments."""
        parser = argparse.ArgumentParser(description="HSI-MAE Configuration")

        # Model
        parser.add_argument("--bands", type=int, default=200)
        parser.add_argument("--encoder-dim", type=int, default=256)
        parser.add_argument("--encoder-layers", type=int, default=4)
        parser.add_argument("--decoder-dim", type=int, default=256)
        parser.add_argument("--patch-size", type=int, default=8)
        parser.add_argument("--mask-ratio", type=float, default=0.75)

        # Data
        parser.add_argument("--data-path", type=str, default="")
        parser.add_argument("--train-split", type=float, default=0.8)
        parser.add_argument("--val-split", type=float, default=0.1)
        parser.add_argument("--normalize", action="store_true", default=True)

        # Training
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--warmup-epochs", type=int, default=10)
        parser.add_argument("--lambda-sam", type=float, default=0.1)

        # System
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--save-dir", type=str, default="checkpoints")
        parser.add_argument("--log-interval", type=int, default=10)
        parser.add_argument("--val-interval", type=int, default=1)

        # Fine-tune
        parser.add_argument("--finetune-mode", type=str, default="full")
        parser.add_argument("--encoder-path", type=str, default="")

        parsed = parser.parse_args(args)

        config = TrainConfig(
            bands=parsed.bands,
            encoder_dim=parsed.encoder_dim,
            encoder_layers=parsed.encoder_layers,
            decoder_dim=parsed.decoder_dim,
            patch_size=parsed.patch_size,
            mask_ratio=parsed.mask_ratio,
            data_path=parsed.data_path,
            train_split=parsed.train_split,
            val_split=parsed.val_split,
            normalize=parsed.normalize,
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            lr=parsed.lr,
            weight_decay=parsed.weight_decay,
            warmup_epochs=parsed.warmup_epochs,
            lambda_sam=parsed.lambda_sam,
            seed=parsed.seed,
            device=parsed.device,
            num_workers=parsed.num_workers,
            save_dir=parsed.save_dir,
            log_interval=parsed.log_interval,
            val_interval=parsed.val_interval,
            finetune_mode=parsed.finetune_mode,
            encoder_path=parsed.encoder_path,
        )
        return cls(config=config)

    @classmethod
    def from_json(cls, path: str | Path) -> ConfigManager:
        """Load config from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        config = TrainConfig(**data)
        return cls(config=config)

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self.config)
