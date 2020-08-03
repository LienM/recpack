import numpy
import pytest
import scipy.sparse

from recpack.metricsv2.recall import RecallK


def test_recallK(X_pred, X_true):
    K = 2
    metric = RecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 2 / 3)
