import numpy as np
from recpack.metricsv2.coverage import Coverage, CoverageK

def test_coverageK(X_pred, X_true):
    K = 2
    metric = CoverageK(K)
    metric.fit(X_true) 

    metric.update(X_pred, X_true)

    assert metric.name == "coverage_2"
    # user 0 gets recommended items 0 and 2
    # user 2 gets recommended items 3 and 4
    # total number of items = 5
    np.testing.assert_almost_equal(metric.value, 4/5)

def test_coverage(X_pred, X_true):
    metric = Coverage()
    metric.fit(X_true) 

    metric.update(X_pred, X_true)

    assert metric.name == "coverage"
    # user 0 gets recommended items 0, 2 and 3
    # user 2 gets recommended items 1, 3 and 4
    # total number of items = 5
    np.testing.assert_almost_equal(metric.value, 1)
