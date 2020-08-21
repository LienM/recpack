from recpack.metrics.diversity import IntraListDiversityK
import numpy as np


def test_ildK(X_true, X_pred, item_features):
    K = 2

    metric = IntraListDiversityK(K)
    metric.fit(item_features)

    metric.calculate(X_true, X_pred)

    np.testing.assert_almost_equal(metric.value, 1 / 3)
