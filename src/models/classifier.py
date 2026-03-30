"""HSI Classifier and Fine-tuner."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import HSIEncoder


class HSIClassifier(nn.Module):
    """
    Linear / MLP classification head for HSI features.

    Maps encoded features [B, dim, H, W] to class logits [B, num_classes, H, W].

    Architecture variants:
        - "linear": single Conv2d (dim → num_classes)
        - "mlp": dim → hidden → num_classes with dropout

    Args:
        in_dim: Input feature dimension (from encoder).
        num_classes: Number of land-cover classes.
        hidden_dim: Hidden dimension for MLP variant.
        variant: Classification head type ("linear" | "mlp").
        dropout: Dropout rate (MLP variant only).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        variant: str = "linear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.variant = variant

        if variant == "linear":
            self.head = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        elif variant == "mlp":
            hid = hidden_dim or in_dim
            self.head = nn.Sequential(
                nn.Conv2d(in_dim, hid, kernel_size=1),
                nn.BatchNorm2d(hid),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Conv2d(hid, num_classes, kernel_size=1),
            )
        else:
            msg = f"Unknown classifier variant: {variant!r} (expected 'linear' or 'mlp')"
            raise ValueError(msg)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify features.

        Args:
            features: Encoded features [B, in_dim, H, W].

        Returns:
            Logits [B, num_classes, H, W].
        """
        return self.head(features)


class HSIFineTuner(nn.Module):
    """
    Full fine-tuning model: Encoder + Classifier.

    Combines a pretrained HSIEncoder with a classification head.
    Supports linear-probe mode (frozen encoder) and full fine-tune mode.

    Args:
        encoder: Pretrained HSIEncoder instance.
        num_classes: Number of land-cover classes.
        classifier_hidden: Hidden dim for MLP classifier (None = linear).
        classifier_variant: "linear" or "mlp".
        freeze_encoder: If True, encoder is frozen at init.
    """

    def __init__(
        self,
        encoder: HSIEncoder,
        num_classes: int,
        classifier_hidden: int | None = None,
        classifier_variant: str = "mlp",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = HSIClassifier(
            in_dim=encoder.get_output_dim(),
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            variant=classifier_variant,
        )

        if freeze_encoder:
            self.encoder.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input HSI.

        Args:
            x: Input [B, bands, H, W].

        Returns:
            Logits [B, num_classes, H, W].
        """
        features = self.encoder(x)
        return self.classifier(features)

    def freeze_encoder(self) -> None:
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze()

    @property
    def encoder_dim(self) -> int:
        return self.encoder.get_output_dim()
