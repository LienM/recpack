import numpy

from recpack.metrics.reciprocal_rank import RRK


def test_rrk(X_pred, X_true):
    K = 2
    metric = RRK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.75)