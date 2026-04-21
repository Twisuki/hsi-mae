"""Visualization utilities for HSI analysis and classification results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# -------------------------------------------------------------------------
# Spectral Signature
# -------------------------------------------------------------------------


def plot_spectral_signature(
    signatures: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Spectral Signatures",
    save_path: str | Path | None = None,
    wavelength_unit: str = "Band",
) -> dict | None:
    """
    Plot spectral signatures (one curve per pixel/material).

    Args:
        signatures: Array of shape [N, C] — N spectra, C bands.
        labels: Optional class labels for coloring (shape [N,]).
        title: Plot title.
        save_path: Optional path to save the figure.
        wavelength_unit: X-axis label prefix (e.g. "nm", "cm⁻¹").

    Returns:
        If matplotlib is available, returns the Axes dict; else None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    num_bands = signatures.shape[1]
    x = np.arange(num_bands)

    fig, ax = plt.subplots(figsize=(10, 5))

    if labels is not None:
        classes = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(classes))
        for i, c in enumerate(classes):
            mask = labels == c
            mean = signatures[mask].mean(axis=0)
            ax.plot(x, mean, label=f"Class {c}", color=cmap(i), linewidth=1.5)
        ax.legend()
    else:
        for i in range(min(signatures.shape[0], 50)):
            ax.plot(x, signatures[i], alpha=0.3, linewidth=0.8)
        ax.plot(x, signatures.mean(axis=0), color="black", linewidth=2, label="Mean")

    ax.set_xlabel(f"{wavelength_unit}")
    ax.set_ylabel("Reflectance (normalized)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return {"fig": fig, "ax": ax}


# -------------------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> dict | None:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Total number of classes.
        class_names: Optional class names for axis labels.
        title: Plot title.
        save_path: Optional path to save the figure.
        normalize: If True, show percentages (0–100%) instead of counts.

    Returns:
        Axes dict if matplotlib available; else None.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix as sk_cm
    except ImportError:
        return None

    cm = sk_cm(y_true, y_pred, labels=list(range(num_classes)))
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm) * 100
        show = cm_norm
        fmt = ".1f"
        cbar_kw = {"label": "Accuracy (%)"}
    else:
        show = cm
        fmt = "d"
        cbar_kw = {"label": "Count"}

    labels_names = class_names or [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes * 0.8)))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=100 if normalize else None)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels_names, rotation=45, ha="right")
    ax.set_yticklabels(labels_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    # Annotate cells
    for i in range(num_classes):
        for j in range(num_classes):
            val = show[i, j]
            color = "white" if val > 50 else "black"
            ax.text(
                j, i, f"{val:{fmt}}", ha="center", va="center", color=color, fontsize=8
            )

    fig.colorbar(im, ax=ax, **cbar_kw)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return {"fig": fig, "ax": ax}


# -------------------------------------------------------------------------
# Classification Map
# -------------------------------------------------------------------------


def plot_classification_map(
    ground_truth: np.ndarray | None,
    prediction: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
    save_path: str | Path | None = None,
) -> dict | None:
    """
    Plot ground truth and prediction maps side by side.

    Args:
        ground_truth: Optional GT map [H, W].
        prediction: Predicted map [H, W].
        num_classes: Total number of classes.
        class_names: Optional class names for colorbar.
        title_gt: Title for ground truth subplot.
        title_pred: Title for prediction subplot.
        save_path: Optional path to save the figure.

    Returns:
        Axes dict if matplotlib available; else None.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return None

    class_names = class_names or [str(i) for i in range(num_classes)]
    cmap = plt.cm.get_cmap("tab10", num_classes)
    cmap_discrete = ListedColormap(cmap.colors[:num_classes])

    n_axes = 2 if ground_truth is not None else 1
    figsize = (5 * n_axes, 4)
    fig, axes = plt.subplots(1, n_axes, figsize=figsize, squeeze=False)

    plots = [(prediction, title_pred)]
    if ground_truth is not None:
        plots.insert(0, (ground_truth, title_gt))

    for ax, (data, name) in zip(axes.flat, plots):
        im = ax.imshow(data, cmap=cmap_discrete, vmin=0, vmax=num_classes)
        ax.set_title(name)
        ax.axis("off")
        ticks = np.arange(num_classes) + 0.5
        cbar = fig.colorbar(im, ax=ax, ticks=ticks, shrink=0.8)
        cbar.ax.set_yticklabels(class_names, fontsize=7)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return {"fig": fig, "axes": axes}


# -------------------------------------------------------------------------
# Training Curve
# -------------------------------------------------------------------------


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_metrics: list[float] | None = None,
    val_metrics: list[float] | None = None,
    metric_name: str = "Metric",
    title: str = "Training Curve",
    save_path: str | Path | None = None,
) -> dict | None:
    """
    Plot training and validation loss/metric curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: Optional list of validation losses per epoch.
        train_metrics: Optional list of training metric values.
        val_metrics: Optional list of validation metric values.
        metric_name: Name of the metric for display.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Axes dict if matplotlib available; else None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_losses, label="Train", linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, label="Val", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Metric
    if train_metrics or val_metrics:
        if train_metrics:
            ax2.plot(epochs, train_metrics, label=f"Train {metric_name}", linewidth=2)
        if val_metrics:
            ax2.plot(epochs, val_metrics, label=f"Val {metric_name}", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(metric_name)
        ax2.set_title(metric_name)
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return {"fig": fig, "ax1": ax1, "ax2": ax2}
