"""Classification metrics for HSI land-cover evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from sklearn.metrics import (
    classification_report as sk_classification_report,
)


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall Accuracy (OA).

    Fraction of correctly classified pixels.
    """
    return accuracy_score(y_true, y_pred)


def average_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """
    Average Accuracy (AA).

    Mean recall across all classes (ignores classes with no samples).
    """
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append((y_true[mask] == y_pred[mask]).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cohen's Kappa coefficient.

    Measures inter-rater agreement accounting for chance.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(max(y_true.max(), y_pred.max()) + 1))
    n = cm.sum()
    if n == 0:
        return 0.0
    sum_observed = cm.diagonal().sum()
    sum_expected = cm.sum(axis=1).dot(cm.sum(axis=0)) / n
    po = sum_observed / n
    pe = sum_expected / n
    if pe == 1.0:
        return 0.0
    return float((po - pe) / (1 - pe))


def f1_scores(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict[str, float]:
    """
    Per-class and macro F1 scores.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Total number of classes.

    Returns:
        Dictionary with "f1_macro" and "f1_class_{i}" for each class.
    """
    f1_per_class: list[float] = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)

    return {
        "f1_macro": float(np.mean(f1_per_class)),
        **{f"f1_class_{i}": float(f) for i, f in enumerate(f1_per_class)},
    }


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> str:
    """
    Generate a formatted classification report.

    Includes per-class precision, recall, F1, and support.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Total number of classes.
        class_names: Optional list of class names.

    Returns:
        Formatted report string.
    """
    labels = list(range(num_classes))
    target_names = class_names or [f"class_{i}" for i in labels]
    return sk_classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0.0,
    )
