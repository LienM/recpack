import numpy
import pytest
import scipy.sparse

import recpack.metrics as metrics


def test_recall(X_pred, X_true):
    K = 2
    metric = metrics.RecallK(K)

    metric.update(X_pred, X_true)

    assert metric.num_users == 2
    assert metric.value == 0.75
