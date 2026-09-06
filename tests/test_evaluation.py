import numpy as np

from taafnet.evaluation import (
    classification_metrics,
    wilson_interval,
)


def test_wilson_interval():
    lower, upper = wilson_interval(
        1219,
        1224,
    )

    assert lower < 1219 / 1224 < upper
    assert 0.990 < lower < 0.991
    assert 0.998 < upper < 0.999


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 3, 4])
    probabilities = np.eye(
        5,
        dtype=float,
    )

    metrics = classification_metrics(
        y_true,
        probabilities,
    )

    assert metrics["Accuracy"] == 1.0
    assert metrics["F1"] == 1.0
