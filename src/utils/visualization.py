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
# Training Curve (Loss + Accuracy in one figure)
# -------------------------------------------------------------------------


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_metrics: list[float] | None = None,
    val_metrics: list[float] | None = None,
    metric_name: str = "Accuracy",
    title: str = "Training Curve",
    save_path: str | Path | None = None,
) -> dict | None:
    """
    Plot training and validation loss/metric curves in one figure.

    Loss and metric are plotted on the same figure with twin y-axes.

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

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss (left y-axis)
    color_loss = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.plot(
        epochs,
        train_losses,
        label="Train Loss",
        color=color_loss,
        linewidth=2,
        linestyle="-",
    )
    if val_losses:
        ax1.plot(
            epochs,
            val_losses,
            label="Val Loss",
            color=color_loss,
            linewidth=2,
            linestyle="--",
        )
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(alpha=0.3)

    # Metric (right y-axis)
    if train_metrics or val_metrics:
        ax2 = ax1.twinx()
        color_metric = "tab:blue"
        ax2.set_ylabel(metric_name, color=color_metric)
        if train_metrics:
            ax2.plot(
                epochs,
                train_metrics,
                label=f"Train {metric_name}",
                color=color_metric,
                linewidth=2,
                linestyle="-",
            )
        if val_metrics:
            ax2.plot(
                epochs,
                val_metrics,
                label=f"Val {metric_name}",
                color=color_metric,
                linewidth=2,
                linestyle="--",
            )
        ax2.tick_params(axis="y", labelcolor=color_metric)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if train_metrics or val_metrics:
        ax2 = fig.axes[1] if len(fig.axes) > 1 else None
        if ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(lines1, labels1, loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return {
        "fig": fig,
        "ax1": ax1,
        "ax2": ax2 if train_metrics or val_metrics else None,
    }


# -------------------------------------------------------------------------
# Reconstruction Progress (multiple epochs in one figure)
# -------------------------------------------------------------------------


def plot_reconstruction_progress(
    originals: list[np.ndarray],
    reconstructions: list[np.ndarray],
    epoch_labels: list[str] | None = None,
    num_bands: int = 200,
    title: str = "Reconstruction Progress",
    save_path: str | Path | None = None,
) -> dict | None:
    """
    Plot reconstruction progress across multiple epochs in one figure.

    Each row shows original spectrum and its reconstruction for one epoch.

    Args:
        originals: List of original spectra [B, C, 1, 1] or [C,], take first sample.
        reconstructions: List of reconstructed spectra same shape as originals.
        epoch_labels: Optional labels for each epoch (e.g., "Epoch 1", "Epoch 10").
        num_bands: Number of spectral bands for x-axis.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Axes dict if matplotlib available; else None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n_epochs = len(originals)
    if n_epochs == 0:
        return None

    # Extract first sample from batch if needed
    originals = [orig[0] if orig.ndim == 4 else orig for orig in originals]
    reconstructions = [
        recon[0] if recon.ndim == 4 else recon for recon in reconstructions
    ]

    # Squeeze extra dimensions
    originals = [orig.squeeze() if orig.ndim > 1 else orig for orig in originals]
    reconstructions = [
        recon.squeeze() if recon.ndim > 1 else recon for recon in reconstructions
    ]

    x = np.arange(num_bands)

    fig, axes = plt.subplots(n_epochs, 2, figsize=(12, 3 * n_epochs), squeeze=False)

    for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
        # Original
        axes[i, 0].plot(x, orig, color="green", linewidth=1.5, label="Original")
        axes[i, 0].set_ylabel(
            f"Epoch {i + 1}" if epoch_labels is None else epoch_labels[i]
        )
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].grid(alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Original")
        if i == n_epochs - 1:
            axes[i, 0].set_xlabel("Band")

        # Reconstruction
        axes[i, 1].plot(
            x, orig, color="green", linewidth=1, alpha=0.5, label="Original"
        )
        axes[i, 1].plot(x, recon, color="red", linewidth=1.5, label="Reconstructed")
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].grid(alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("Reconstruction")
        if i == n_epochs - 1:
            axes[i, 1].set_xlabel("Band")
        if i == 0:
            axes[i, 1].legend(fontsize=8)

    fig.suptitle(title, y=1.0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return {"fig": fig, "axes": axes}


# -------------------------------------------------------------------------
# Confusion Matrix (standalone, already implemented above)
# -------------------------------------------------------------------------
