# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.tests.test_algorithms.conftest import ITEM_IX, TIMESTAMP_IX, USER_IX

from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd

from recpack.matrix import InteractionMatrix
from recpack.scenarios import LastItemPrediction
from recpack.algorithms.p2v import Prod2Vec


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
        distribution="unigram",
    )
    prod._init_model(mat)
    prod.model_.input_embeddings = p2v_embedding

    prod.save = MagicMock(return_value=True)
    return prod


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
