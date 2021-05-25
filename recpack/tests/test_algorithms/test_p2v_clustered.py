from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch

from recpack.algorithms.p2v_clustered import Prod2VecClustered
from recpack.data.matrix import InteractionMatrix

from recpack.tests.test_algorithms.util import assert_changed


@pytest.fixture(scope="function")
def prod2vec(p2v_embedding, mat):
    prod = Prod2VecClustered(
        embedding_size=50,
        num_neg_samples=2,
        window_size=2,
        stopping_criterion="precision",
        K=5,
        batch_size=5,
        learning_rate=0.01,
        max_epochs=10,
        stop_early=False,
        max_iter_no_change=5,
        min_improvement=0.0001,
        save_best_to_file=False,
        replace=False,
        exact=True,
        keep_last=True,
        num_clusters=4,
        Kcl=2,
    )
    prod._init_model(mat)
    prod.model_.input_embeddings = p2v_embedding

    prod.save = MagicMock(return_value=True)
    return prod


def test__predict(prod2vec, larger_mat):
    prod2vec._init_model(larger_mat)
    matrix = csr_matrix((6, 25))
    matrix[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 1

    prod2vec._create_similarity_matrix(larger_mat)
    predictions = prod2vec._batch_predict(matrix)


def test_cluster_similarity_computation():
    alg = Prod2VecClustered(
        embedding_size=2,
        num_neg_samples=2,
        window_size=2,
        stopping_criterion="precision",
        K=5,
        batch_size=5,
        learning_rate=0.01,
        max_epochs=10,
        stop_early=False,
        max_iter_no_change=5,
        min_improvement=0.0001,
        save_best_to_file=False,
        replace=False,
        exact=True,
        keep_last=True,
        num_clusters=4,
        Kcl=2,
    )

    # assume we fitted a very easy to cluster set of embeddings.
    alg.model_ = MagicMock()
    alg.model_.input_embeddings = MagicMock()
    alg.model_.input_embeddings.weight = torch.Tensor(
        [
            [-1, -1],
            [-1.2, -1.2],
            [-1, 1],
            [-1.2, 1.2],
            [1, 1],
            [1.2, 1.2],
            [1, -1],
            [1.2, -1.2],
        ]
    )

    clusters = alg._create_item_to_cluster_array()
    assert clusters.shape[0] == 8
    assert 0 in clusters
    assert 1 in clusters
    assert 2 in clusters
    assert 3 in clusters

    assert clusters.max() == 3

    users = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    items = [0, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 0, 8, 1]
    ts = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    # Special data matrix, which causes cluster 1 -> 1 and 2, 2 -> 2 and 3 , ...
    # fmt:off
    df = pd.DataFrame.from_dict(
        {
            "user": [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7],
            "item": [0, 1, 2, 1, 3, 2, 3, 4, 3, 5, 4, 5, 6, 5, 7, 6, 7, 0, 7, 1],
            "ts":   [1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2],
        }
    )
    # fmt:on

    im = InteractionMatrix(df, "item", "user", timestamp_ix="ts")

    c2c = alg._create_cluster_to_cluster_neighborhoods(im, clusters)

    # Because clusters are not assigned in order,
    # we need to get them based on their item values
    # Remember that we created the embeddings such that clusters are:
    # (0, 1), (2, 3), (4, 5), (6, 7)
    c_0 = clusters[0]
    c_1 = clusters[2]
    c_2 = clusters[4]
    c_3 = clusters[6]

    assert c2c.shape == (4, 4)

    np.testing.assert_array_equal(c2c[c_0].nonzero()[1], np.array(sorted([c_0, c_1])))
    np.testing.assert_array_equal(c2c[c_1].nonzero()[1], np.array(sorted([c_1, c_2])))
    np.testing.assert_array_equal(c2c[c_2].nonzero()[1], np.array(sorted([c_2, c_3])))
    np.testing.assert_array_equal(c2c[c_3].nonzero()[1], np.array(sorted([c_3, c_0])))


def test_training_epoch(prod2vec, mat):

    prod2vec._init_model(mat)

    params = [o for o in prod2vec.model_.named_parameters() if o[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    prod2vec._train_epoch(mat)

    device = prod2vec.device

    assert_changed(params_before, params, device)


def test_predict_warning(prod2vec, mat):
    with pytest.warns(UserWarning, match="K is larger than the number of items."):
        prod2vec._create_similarity_matrix(mat)


# def test_overfit(prod2vec):
#     prod2vec.max_epochs = 200
#     data = {
#         "user": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
#         "item": [0, 1, 2, 0, 1, 0, 1, 2, 0, 2, 3, 4, 5, 3, 4, 3, 4, 5, 3, 5],
#         "timestamp": list(range(0, 20))
#     }
#     df = pd.DataFrame.from_dict(data)
#     im = InteractionMatrix(df, "item", "user", timestamp_ix="timestamp")
#
#     # For user 0 we should learn to predict 0 -> 1
#     # For user 1 we should learn to predict 0 -> 2
#     # For user 2 we should learn to predict 3 -> 4
#     # For user 3 we should learn to predict 3 -> 5
#     scenario = NextItemPrediction(validation=True)
#     scenario.split(im)
#     train = scenario.train_X
#     val_data_in = scenario._validation_data_in
#     val_data_out = scenario._validation_data_out
#
#     prod2vec._init_model(train)
#
#     # Overfitting to make sure we get "deterministic" results
#     prod2vec.fit(train, (val_data_in, val_data_out))
#
#     similarity_matrix = prod2vec.similarity_matrix_.toarray()
#
#     print(similarity_matrix)
#     # Get the most similar item for each item
#     max_similarity_items = np.argmax(similarity_matrix, axis=1)
#     # 3,4,5 should be close together in the vectors space -> 0,1,2 shouldn't be the most similar to either 3,4,5
#     assert 0 not in max_similarity_items[3:]
#     assert 1 not in max_similarity_items[3:]
#     assert 2 not in max_similarity_items[3:]
#     # 0,1,2 should be close together in the vectors space -> 3,4,5 shouldn't be the most similar to either 0,1,2
#     assert 3 not in max_similarity_items[:3]
#     assert 4 not in max_similarity_items[:3]
#     assert 5 not in max_similarity_items[:3]