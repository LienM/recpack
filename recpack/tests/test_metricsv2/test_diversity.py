from recpack.metricsv2.diversity import IntraListDiversity, IntraListDiversityK
import numpy as np


def test_ild(X_pred, item_features):
    metric = IntraListDiversity()
    metric.fit(item_features)

    metric.calculate(None, X_pred)
    print(X_pred)

    assert metric.results.shape[0] == 2

    print(metric.results)
    np.testing.assert_almost_equal(metric.value, 2/3)


def test_ildK(X_pred, item_features):
    K = 2

    metric = IntraListDiversityK(K)
    metric.fit(item_features)

    metric.calculate(None, X_pred)

    assert metric.results.shape[0] == 2

    print(metric.results)
    np.testing.assert_almost_equal(metric.value, 0.5)
