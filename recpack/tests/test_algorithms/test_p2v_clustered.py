# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch

from recpack.algorithms.p2v_clustered import Prod2VecClustered
from recpack.algorithms.util import get_users
from recpack.matrix import InteractionMatrix

from recpack.tests.test_algorithms.util import assert_changed


@pytest.fixture(scope="function")
def prod2vec(p2v_embedding, mat):
    prod = Prod2VecClustered(
        num_components=50,
        num_negatives=2,
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
    predictions = prod2vec._batch_predict(matrix, get_users(matrix))

    assert predictions.shape == matrix.shape


def test_cluster_similarity_computation():
    alg = Prod2VecClustered(
        num_components=2,
        num_negatives=2,
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

    # Check if the correct number of clusters was created
    cluster_assignments = alg._cluster(alg.model_.input_embeddings.weight.cpu().detach().numpy())
    assert cluster_assignments.shape[0] == 8
    assert 0 in cluster_assignments
    assert 1 in cluster_assignments
    assert 2 in cluster_assignments
    assert 3 in cluster_assignments

    assert cluster_assignments.max() == 3

    # Data Matrix is constructed so that
    # neighbours(c_0) = [c_0, c_1]
    # neighbours(c_1) = [c_1, c_2]
    # etc. as in the test at the bottom.
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

    c2c = alg._get_top_K_clusters(im, cluster_assignments)

    # Because clusters are not assigned in order,
    # we need to get them based on their item values
    # Remember that we created the embeddings such that clusters are:
    # (0, 1), (2, 3), (4, 5), (6, 7)
    c_0 = cluster_assignments[0]
    c_1 = cluster_assignments[2]
    c_2 = cluster_assignments[4]
    c_3 = cluster_assignments[6]

    assert c2c.shape == (4, 4)

    np.testing.assert_array_equal(c2c[c_0].nonzero()[1], np.array(sorted([c_0, c_1])))
    np.testing.assert_array_equal(c2c[c_1].nonzero()[1], np.array(sorted([c_1, c_2])))
    np.testing.assert_array_equal(c2c[c_2].nonzero()[1], np.array(sorted([c_2, c_3])))
    np.testing.assert_array_equal(c2c[c_3].nonzero()[1], np.array(sorted([c_3, c_0])))

    # Check if only items from neighbouring clusters are nonzero

    alg._create_similarity_matrix(im)

    # All items in neighbouring clusters except itself.
    np.testing.assert_array_equal([1, 2, 3], alg.similarity_matrix_[0, :].nonzero()[1])
    np.testing.assert_array_equal([3, 4, 5], alg.similarity_matrix_[2, :].nonzero()[1])
    np.testing.assert_array_equal([5, 6, 7], alg.similarity_matrix_[4, :].nonzero()[1])
    np.testing.assert_array_equal([0, 1, 7], alg.similarity_matrix_[6, :].nonzero()[1])


def test_training_epoch(prod2vec, mat):

    prod2vec._init_model(mat)

    params = [o for o in prod2vec.model_.named_parameters() if o[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    with pytest.warns(UserWarning, match="There are clusters without neighbours"):
        prod2vec._train_epoch(mat)

    device = prod2vec.device

    assert_changed(params_before, params, device)


def test_predict_warning(prod2vec, mat):
    with pytest.warns(UserWarning, match="K is larger than the number of items."):
        prod2vec._create_similarity_matrix(mat)


def test_fit_no_interaction_matrix(prod2vec, mat):
    with pytest.raises(TypeError):
        prod2vec.fit(mat.binary_values, (mat, mat))


def test_fit_no_timestamps(prod2vec, mat):
    with pytest.raises(ValueError):
        prod2vec.fit(mat.eliminate_timestamps(), (mat, mat))


def test_no_self_similarity(prod2vec, mat):

    prod2vec._create_similarity_matrix(mat)
    similarity_matrix = prod2vec.similarity_matrix_.toarray()
    assert (similarity_matrix[np.arange(mat.shape[1]), np.arange(mat.shape[1])] == 0).all()


@pytest.mark.parametrize(
    "active_items, inactive_items",
    [
        ([0, 1, 2, 3], [4]),
        ([0, 2, 3], [1, 4]),
    ],
)
def test_no_similarity_from_inactive_items(prod2vec, mat, active_items, inactive_items):
    matrix = csr_matrix((mat.shape[1], mat.shape[1]))
    matrix[active_items, active_items] = 1

    prod2vec._create_similarity_matrix(mat.items_in(active_items))
    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    assert (similarity_matrix[inactive_items] == 0).all()


@pytest.mark.parametrize(
    "active_items, inactive_items",
    [
        ([0, 1, 2, 3], [4]),
        ([0, 2, 3], [1, 4]),
    ],
)
def test_no_similarity_to_inactive_items(prod2vec, mat, active_items, inactive_items):
    matrix = csr_matrix((mat.shape[1], mat.shape[1]))
    matrix[active_items, active_items] = 1

    prod2vec._create_similarity_matrix(mat.items_in(active_items))
    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    assert (similarity_matrix[:, inactive_items] == 0).all()
