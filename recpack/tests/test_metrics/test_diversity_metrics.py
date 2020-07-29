from recpack.metrics.diversity_metrics import IntraListDistanceK
from recpack.algorithms.user_item_interactions_algorithms.nn_algorithms import ItemKNN


def test_get_ild(data):
    prediction_1 = [0, 2, 3]
    prediction_2 = [0, 1, 3]

    ild_metric = IntraListDistanceK(2)
    ild_metric.fit(data)

    # prediction 1 is less diverse compared to prediction 2 
    ild_1 = ild_metric._get_ild(prediction_1, data.shape[1])
    ild_2 = ild_metric._get_ild(prediction_2, data.shape[1])

    assert ild_2 < ild_1

def test_intra_list_distance_K(data, X_pred, X_true):
    K = 2

    ild_metric = IntraListDistanceK(K)
    ild_metric.fit(data)

    ild_metric.update(X_pred, X_true)

    assert len(ild_metric.distances) == 2
