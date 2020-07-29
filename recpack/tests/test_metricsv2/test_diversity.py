from recpack.metricsv2.diversity import IntraListDiversity, IntraListDiversityK
import numpy as np


def test_ild(X_pred, item_features):
    metric = IntraListDiversity()
    metric.fit(item_features)

    metric.update(X_pred, None)
    print(X_pred)

    assert metric.name == "intra_list_diversity"
    assert metric.results.shape[0] == 2

    print(metric.results)
    np.testing.assert_almost_equal(metric.value, 2/3)


def test_ildK(X_pred, item_features):
    K = 2

    metric = IntraListDiversityK(K)
    metric.fit(item_features)

    metric.update(X_pred, None)

    assert metric.name == "intra_list_diversity_2"
    assert metric.results.shape[0] == 2

    print(metric.results)
    np.testing.assert_almost_equal(metric.value, 0.5)
