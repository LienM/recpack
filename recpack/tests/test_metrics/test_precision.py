import numpy
import pytest
import scipy.sparse

from recpack.metrics.precision import PrecisionK


def test_precisionK(X_pred, X_true):
    K = 2
    metric = PrecisionK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.75)
