import numpy
import pytest
import scipy.sparse

from recpack.metricsv2.reciprocal_rank import RRK


def test_rrk(X_pred, X_true):
    K = 2
    metric = RRK(K)

    metric.calculate(X_true, X_pred)

    assert metric.results.shape[0] == 2
    print(metric.scores_.toarray())
    numpy.testing.assert_almost_equal(metric.value, 0.75)
