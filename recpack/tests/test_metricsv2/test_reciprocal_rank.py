import numpy
import pytest
import scipy.sparse

from recpack.metricsv2.reciprocal_rank import RR


def test_rr(X_pred, X_true):
    metric = RR()

    metric.calculate(X_true, X_pred)

    assert metric.results.shape[0] == 2
    numpy.testing.assert_almost_equal(metric.value, 0.75)
