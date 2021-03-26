import numpy as np
import pytest

from recpack.metrics import HitK, WeightedHitK, DiscountedGainK


def test_hit_K(X_true, X_pred):

    K = 2
    metric = HitK(K=K)

    metric.calculate(X_true, X_pred)

    assert metric.value == (2 + 1) / 2

    # Row per user and prediction in top K
    assert metric.results.shape == (4, 3)

    assert metric.results.score.sum() == 3


def test_weighted_hit_K(X_true, X_pred):

    K = 2
    metric = WeightedHitK(K=K)

    metric.calculate(X_true, X_pred)

    print(metric.results)
    assert metric.value == (2 / 2 + 1 / 3) / 2

    # Row per user and prediction in top K
    assert metric.results.shape == (4, 3)

    assert metric.results.score.sum() == (2 / 2) + (1 / 3)


def test_discounted_gain_K(X_true, X_pred):

    K = 2
    metric = DiscountedGainK(K=K)

    metric.calculate(X_true, X_pred)

    print(metric.results)
    assert metric.value == (1 + (1 / np.log2(3)) + (1 / np.log2(3))) / 2

    # Row per user and prediction in top K
    assert metric.results.shape == (4, 3)

    assert metric.results.score.sum() == 1 + (1 / np.log2(3)) + (1 / np.log2(3))
