# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.tests.test_algorithms.conftest import ITEM_IX, TIMESTAMP_IX, USER_IX
import tempfile
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recpack.algorithms.p2v import Prod2Vec, window
from recpack.matrix import InteractionMatrix
from recpack.scenarios import LastItemPrediction
from recpack.matrix import to_csr_matrix
from recpack.algorithms.util import get_users
from recpack.tests.test_algorithms.util import assert_changed, assert_same


@pytest.fixture(scope="function")
def prod2vec(p2v_embedding, mat):
    prod = Prod2Vec(
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
    )
    prod._init_model(mat)
    prod.model_.input_embeddings = p2v_embedding

    prod.save = MagicMock(return_value=True)
    return prod


@pytest.fixture(scope="module")
def diagonal_interaction_matrix() -> InteractionMatrix:
    matrix = csr_matrix((6, 5))
    matrix[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 1

    return InteractionMatrix.from_csr_matrix(matrix)


def test_window():
    # TODO what about the error for small values?

    sequence = [
        (123, ["computer", "artificial", "intelligence", "dog", "trees"]),
        (145, ["human", "intelligence", "cpu", "graph"]),
        (1, ["intelligence"]),
        (3, ["artificial", "intelligence", "system"]),
    ]
    # Create a window of size 3:
    # sequence 1: 5 windows
    # sequence 2: 4 windows
    # sequence 3: 1 window
    # sequence 4: 3 window
    # => 13 windows
    windowed_seq = window(sequence, 1)

    row, col = windowed_seq.shape
    assert row == 13
    assert col == 3


def test_training_epoch(prod2vec, mat):
    prod2vec._init_model(mat)

    params = [o for o in prod2vec.model_.named_parameters() if o[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    prod2vec._train_epoch(mat)

    device = prod2vec.device

    assert_changed(params_before, params, device)


def test_evaluation_epoch(prod2vec, mat):
    prod2vec._init_model(mat)

    params = [o for o in prod2vec.model_.named_parameters() if o[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    prod2vec.best_model = tempfile.NamedTemporaryFile()
    prod2vec._create_similarity_matrix(mat)
    # run a training step
    prod2vec._evaluate(to_csr_matrix(mat), to_csr_matrix(mat))

    device = prod2vec.device

    assert_same(params_before, params, device)
    prod2vec.best_model.close()


def test_predict_warning(prod2vec, diagonal_interaction_matrix):
    prod2vec.K = diagonal_interaction_matrix.shape[1] + 10
    with pytest.warns(UserWarning, match="K was set to a value larger than the number of items"):
        prod2vec._create_similarity_matrix(diagonal_interaction_matrix)


def test_create_similarity_matrix(prod2vec, diagonal_interaction_matrix):
    # TODO Keep test inputs and outputs together
    prod2vec._create_similarity_matrix(diagonal_interaction_matrix)
    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    # Get the most similar item for each item
    np.testing.assert_array_equal(np.argmax(similarity_matrix, axis=1), [1, 0, 3, 2, 0])

    # Cosine similarities can be calculated by hand easily
    # Only get the most similar item using max
    assert max(similarity_matrix[0]) == pytest.approx(0.98473, 0.00005)
    assert max(similarity_matrix[2]) == pytest.approx(0.5, 0.00005)
    assert max(similarity_matrix[4]) == pytest.approx(0.70711, 0.00005)


def test_create_similarity_matrix_no_self_similarity(prod2vec, diagonal_interaction_matrix):
    prod2vec._create_similarity_matrix(diagonal_interaction_matrix)
    similarity_matrix = prod2vec.similarity_matrix_.toarray()
    assert (similarity_matrix[np.arange(5), np.arange(5)] == 0).all()


@pytest.mark.parametrize(
    "active_items, inactive_items",
    [
        ([0, 1, 2, 3], [4]),
        ([0, 2, 3], [1, 4]),
        ([1, 4], [0, 2, 3]),
        ([1], [0, 2, 3, 4]),
    ],
)
def test_create_similarity_matrix_no_similarities_from_inactive_items(prod2vec, active_items, inactive_items):
    matrix = csr_matrix((6, 5))
    matrix[active_items, active_items] = 1

    prod2vec._create_similarity_matrix(InteractionMatrix.from_csr_matrix(matrix))
    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    assert (similarity_matrix[inactive_items] == 0).all()


@pytest.mark.parametrize(
    "active_items, inactive_items",
    [
        ([0, 1, 2, 3], [4]),
        ([0, 2, 3], [1, 4]),
        ([1, 4], [0, 2, 3]),
        ([1], [0, 2, 3, 4]),
    ],
)
def test_create_similarity_matrix_no_similarities_to_inactive_items(prod2vec, active_items, inactive_items):
    matrix = csr_matrix((6, 5))
    matrix[active_items, active_items] = 1

    prod2vec._create_similarity_matrix(InteractionMatrix.from_csr_matrix(matrix))
    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    assert (similarity_matrix[:, inactive_items] == 0).all()


def test_batch_predict(prod2vec):
    # Rewrite with different matrix
    matrix = csr_matrix((6, 5))
    matrix[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 1

    prod2vec._create_similarity_matrix(InteractionMatrix.from_csr_matrix(matrix))
    predictions = prod2vec._batch_predict(matrix, get_users(matrix))

    np.testing.assert_array_almost_equal(
        predictions.toarray(),
        np.array(
            [
                [0.0, 0.98473193, 0.0, 0.0, 0.70710678],
                [0.98473193, 0.0, 0.0, 0.12309149, 0.69631062],
                [0.0, 0.0, 0.0, 0.5, 0.0],
                [0.0, 0.12309149, 0.5, 0.0, 0.0],
                [0.70710678, 0.69631062, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_skipgram_sample_pairs_large_sample(prod2vec, larger_mat):
    prod2vec._init_model(larger_mat)

    U = prod2vec.num_negatives

    for item_1, item_2, neg_items in prod2vec._skipgram_sample_pairs(larger_mat):
        assert item_1.shape[0] == item_2.shape[0]
        assert item_1.shape[0] == neg_items.shape[0]
        assert neg_items.shape[1] == U

        # There should be no collisions between columns of negative samples
        for i in range(U):
            for j in range(i):
                overlap = neg_items[:, j].numpy() == neg_items[:, i].numpy()

                np.testing.assert_array_equal(overlap, False)


def test_skipgram_sample_pairs_error(prod2vec):
    data = {
        TIMESTAMP_IX: [3, 2, 1, 4, 0, 1, 2, 4, 0, 1, 2],
        ITEM_IX: [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2],
        USER_IX: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
    }
    df = pd.DataFrame.from_dict(data)

    matrix = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    prod2vec._init_model(matrix)

    # Too few possible negatives
    with pytest.raises(ValueError):
        for item_1, item_2, neg_items in prod2vec._skipgram_sample_pairs(matrix):
            pass


def test_skipgram_sample_pairs_small_sample(prod2vec, mat):
    prod2vec._init_model(mat)

    all_item_1 = np.array([], dtype=int)
    all_item_2 = np.array([], dtype=int)
    all_neg_items = None

    for item_1, item_2, neg_items in prod2vec._skipgram_sample_pairs(mat):
        all_item_1 = np.append(all_item_1, item_1)
        all_item_2 = np.append(all_item_2, item_2)

        if all_neg_items is None:
            all_neg_items = neg_items
        else:
            all_neg_items = np.vstack([all_neg_items, neg_items])

    generated_positive_pairs = np.column_stack([all_item_1, all_item_2])

    expected_positive_pairs = np.array(
        [[1, 0], [0, 1], [2, 3], [3, 2], [1, 0], [0, 1], [4, 2], [2, 4], [0, 1], [1, 0]]
    )

    sorted_generated_positive_pairs = list(map(tuple, generated_positive_pairs))
    sorted_generated_positive_pairs.sort()
    sorted_expected_positive_pairs = list(map(tuple, expected_positive_pairs))
    sorted_expected_positive_pairs.sort()

    np.testing.assert_array_equal(sorted_generated_positive_pairs, sorted_expected_positive_pairs)


def test_overfit(prod2vec):
    prod2vec.max_epochs = 10
    prod2vec.learning_rate = 0.1
    data = {
        "user": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "item": [0, 1, 2, 0, 1, 0, 1, 2, 0, 2, 3, 4, 5, 3, 4, 3, 4, 5, 3, 5],
        "timestamp": list(range(0, 20)),
    }
    df = pd.DataFrame.from_dict(data)
    im = InteractionMatrix(df, "item", "user", timestamp_ix="timestamp")

    # For user 0 we should learn to predict 0 -> 1
    # For user 1 we should learn to predict 0 -> 2
    # For user 2 we should learn to predict 3 -> 4
    # For user 3 we should learn to predict 3 -> 5
    scenario = LastItemPrediction(validation=True)
    scenario.split(im)
    train = scenario.validation_training_data
    val_data_in = scenario._validation_data_in
    val_data_out = scenario._validation_data_out

    prod2vec._init_model(train)

    # Overfitting to make sure we get "deterministic" results
    prod2vec.fit(train, (val_data_in, val_data_out))

    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    # Get the most similar item for each item
    max_similarity_items = np.argmax(similarity_matrix, axis=1)
    # 3,4,5 should be close together in the vectors space -> 0,1,2 shouldn't be the most similar to either 3,4,5
    assert 0 not in max_similarity_items[3:]
    assert 1 not in max_similarity_items[3:]
    assert 2 not in max_similarity_items[3:]
    # 0,1,2 should be close together in the vectors space -> 3,4,5 shouldn't be the most similar to either 0,1,2
    assert 3 not in max_similarity_items[:3]
    assert 4 not in max_similarity_items[:3]
    assert 5 not in max_similarity_items[:3]


def test_p2v_fit_no_interaction_matrix(prod2vec, mat):
    with pytest.raises(TypeError):
        prod2vec.fit(mat.binary_values, (mat, mat))


def test_p2v_fit_no_timestamps(prod2vec, mat):
    with pytest.raises(ValueError):
        prod2vec.fit(mat.eliminate_timestamps(), (mat, mat))
