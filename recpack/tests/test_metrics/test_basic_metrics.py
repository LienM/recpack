import numpy
import pytest
import scipy.sparse

import recpack.metrics as metrics


def test_recall(X_pred, X_true):
    K = 2
    metric = metrics.RecallK(K)

    metric.update(X_pred, X_true)

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, 0.75)


def test_mrr(X_pred, X_true):
    K = 2
    metric = metrics.MeanReciprocalRankK(K)

    metric.update(X_pred, X_true)

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, 0.75)


def test_ndcg(X_pred, X_true_simplified):
    K = 2
    metric = metrics.NDCGK(K)

    metric.update(X_pred, X_true_simplified)

    #  Number of true items for each user is one.
    IDCG = sum(1 / numpy.log2((i + 1) + 1) for i in range(0, 1))

    expected_value = (
        (1 / numpy.log2(2 + 1)) / IDCG +  # Predicted item in position "2" count from 1.
        (1 / numpy.log2(2 + 1)) / IDCG  # Predicted item in position "2" count from 1.
    ) / 2  # 2 users

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_value)
