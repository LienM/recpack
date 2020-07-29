import numpy
import pytest
import scipy.sparse

from recpack.metricsv2.recall import Recall, RecallK

def test_recallK(X_pred, X_true):
    K = 2
    metric = RecallK(K)

    metric.calculate(X_true, X_pred)

    assert metric.results.shape[0] == 2
    numpy.testing.assert_almost_equal(metric.value, 0.75)

def test_recall(X_pred, X_true):
    metric = Recall()

    metric.calculate(X_true, X_pred)

    assert metric.results.shape[0] == 2
    numpy.testing.assert_almost_equal(metric.value, 5/6) # Average of 1 and 2/3
