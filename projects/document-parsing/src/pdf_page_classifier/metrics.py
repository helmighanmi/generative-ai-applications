# Path: projects/document-parsing/src/pdf_page_classifier/metrics.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Classification metrics for offline evaluation.

The helpers in this module are intentionally small and free of project-specific
logic so they can be tested against hand-computed values. They depend only on
numpy and scikit-learn, both already required by the classifier.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DEFAULT_LABELS: tuple[str, ...] = ("document", "diagram")


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision, recall, F1 and support for a single label."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Aggregate metrics for a set of predictions."""

    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion: np.ndarray
    labels: tuple[str, ...]
    n_samples: int


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> ClassificationMetrics:
    """Compute accuracy, macro F1, per-class metrics and the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels, aligned with ``y_true``.
        labels: Label order used for per-class metrics and the confusion matrix.

    Raises:
        ValueError: If the inputs are empty, differ in length, or contain a
            label outside ``labels``.
    """
    truth = list(y_true)
    predicted = list(y_pred)
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    if not truth:
        raise ValueError("metrics require at least one sample")

    known = set(labels)
    unknown = (set(truth) | set(predicted)) - known
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(labels),
        zero_division=0,
    )
    per_class = {
        label: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(labels)
    }

    matrix = confusion_matrix(truth, predicted, labels=list(labels))
    accuracy = float(np.mean(np.asarray(truth) == np.asarray(predicted)))
    macro_f1 = float(np.mean(f1))

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=matrix,
        labels=tuple(labels),
        n_samples=len(truth),
    )


def bootstrap_accuracy_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for accuracy.

    The sample size in this project is small, so a point estimate alone would
    overstate precision. The resampling seed is fixed to keep reports
    reproducible.
    """
    truth = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    if truth.size == 0:
        raise ValueError("bootstrap requires at least one sample")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, truth.size, size=(n_resamples, truth.size))
    accuracies = np.mean(truth[indices] == predicted[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(accuracies, [tail, 1.0 - tail])
    return float(lower), float(upper)


def assert_disjoint_splits(
    calibration_ids: Iterable[str],
    test_ids: Iterable[str],
) -> None:
    """Fail loudly when a test case is also used for calibration.

    Calibrating on pages that later appear in the test set inflates metrics and
    invalidates the reported numbers, so the evaluation treats overlap as an
    error rather than a warning.
    """
    overlap = set(calibration_ids) & set(test_ids)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"calibration and test splits overlap: {joined}")
