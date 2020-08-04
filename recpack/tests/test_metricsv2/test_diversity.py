from recpack.metricsv2.diversity import IntraListDiversityK
import numpy as np


def test_ildK(X_true, X_pred, item_features):
    K = 2

    metric = IntraListDiversityK(K)
    metric.fit(item_features)

    metric.calculate(X_true, X_pred)

    assert X_true.shape == (10, 5)

    assert metric.results.shape[0] == 2

    print(metric.results)
    np.testing.assert_almost_equal(metric.value, 0.5)
