import numpy

from recpack.metrics.recall import RecallK, CalibratedRecallK


def test_recallK(X_pred, X_true):
    K = 2
    metric = RecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 2 / 3)

    K = 1
    metric = RecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.25)


def test_calibrated_recallK(X_pred, X_true):
    K = 2
    metric = CalibratedRecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.75)

    K = 1
    metric = CalibratedRecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.5)