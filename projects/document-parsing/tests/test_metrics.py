from __future__ import annotations

import numpy as np
import pytest
from pdf_page_classifier.metrics import (
    assert_disjoint_splits,
    bootstrap_accuracy_ci,
    compute_metrics,
)


def test_metrics_match_hand_computed_values() -> None:
    metrics = compute_metrics(
        ["document", "document", "diagram", "diagram"],
        ["document", "diagram", "diagram", "diagram"],
    )

    assert metrics.n_samples == 4
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics.per_class["document"].precision == pytest.approx(1.0)
    assert metrics.per_class["document"].recall == pytest.approx(0.5)
    assert metrics.per_class["diagram"].precision == pytest.approx(2 / 3)
    assert metrics.per_class["diagram"].recall == pytest.approx(1.0)
    assert metrics.per_class["document"].support == 2
    assert metrics.per_class["diagram"].support == 2


def test_confusion_matrix_rows_are_actual_labels() -> None:
    metrics = compute_metrics(
        ["document", "document", "diagram", "diagram"],
        ["document", "diagram", "diagram", "diagram"],
    )

    assert metrics.labels == ("document", "diagram")
    np.testing.assert_array_equal(metrics.confusion, [[1, 1], [0, 2]])


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    truth = ["document", "diagram", "document", "diagram", "document"]
    predicted = ["document", "diagram", "diagram", "diagram", "document"]
    accuracy = float(np.mean(np.asarray(truth) == np.asarray(predicted)))

    lower, upper = bootstrap_accuracy_ci(truth, predicted)

    assert 0.0 <= lower <= accuracy <= upper <= 1.0


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    truth = ["document", "diagram", "document", "diagram", "document", "diagram"]
    predicted = ["document", "document", "document", "diagram", "diagram", "diagram"]

    first = bootstrap_accuracy_ci(truth, predicted, seed=7)
    second = bootstrap_accuracy_ci(truth, predicted, seed=7)

    assert first == second


def test_overlapping_splits_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits(["a", "b"], ["b", "c"])

    assert_disjoint_splits(["a", "b"], ["c", "d"])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_metrics(["document"], ["document", "diagram"])


def test_unknown_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        compute_metrics(["chart"], ["diagram"])
