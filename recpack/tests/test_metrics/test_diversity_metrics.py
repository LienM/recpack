from recpack.metrics.diversity_metrics import IntraListDistanceK
from recpack.algorithms.user_item_interactions_algorithms.nn_algorithms import ItemKNN


def test_get_idl(data):
    sim_algo = ItemKNN(2, normalise=True)
    sim_algo.fit(data)

    prediction_1 = [0, 2]
    prediction_2 = [0, 1]

    idl_metric = IntraListDistanceK(sim_algo, 2)

    idl_1 = idl_metric._get_idl(prediction_1, data.shape[1])
    idl_2 = idl_metric._get_idl(prediction_2, data.shape[1])

    assert idl_1 < idl_2

def test_intra_list_distance_K(data, X_pred, X_true):
    K = 2
    sim_algo = ItemKNN(K, normalise=True)
    sim_algo.fit(data)

    metric = IntraListDistanceK(sim_algo, K)

    metric.update(X_pred, X_true)

    assert metric.number_of_users == 2
