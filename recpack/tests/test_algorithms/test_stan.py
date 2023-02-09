# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms.stan import STAN
from recpack.matrix import InteractionMatrix


@pytest.fixture()
def algo():
    return STAN(
        K=2,
        interaction_decay=1.0,
        session_decay=1 / 5,
        distance_from_match_decay=1.0,
    )


@pytest.fixture()
def mini_dataset():
    df = pd.DataFrame.from_dict(
        {
            "uid": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "iid": [0, 1, 2, 2, 3, 1, 3, 2, 3],
            "tst": [1, 2, 3, 1, 2, 4, 5, 6, 7],
        }
    )
    return InteractionMatrix(df, user_ix="uid", item_ix="iid", timestamp_ix="tst")


@pytest.fixture()
def mini_training_dataset(mini_dataset):
    return mini_dataset.users_in([0, 1])


@pytest.fixture()
def mini_test_dataset(mini_dataset):
    return mini_dataset.users_in([2, 3])


def test_init(algo):
    assert algo.identifier == "STAN(K=2,distance_from_match_decay=1.0," "interaction_decay=1.0,session_decay=0.2)"


def test_fit(algo, mini_training_dataset):

    algo.fit(mini_training_dataset)

    expected_interaction_positions = np.array([[1, 2, 3, 0], [0, 0, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(
        algo.session_interactions_positions_.toarray(),
        expected_interaction_positions,
    )

    expected_session_timestamps = [[3, 2, 0, 0]]
    np.testing.assert_array_equal(algo.historical_session_timestamps_.A.T, expected_session_timestamps)


def test_compute_session_similarity(algo, mini_training_dataset, mini_test_dataset):
    algo.fit(mini_training_dataset)

    sim = algo._compute_session_similarity(mini_test_dataset.last_timestamps_matrix)
    expected_similarity = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [np.exp(-1) / np.sqrt(2 * 3), 1 / np.sqrt(2 * 2), 0, 0],
        [
            np.exp(-1) * 1 / (np.sqrt(2 * 3)),
            (1 + np.exp(-1)) / np.sqrt(2 * 2),
            0,
            0,
        ],
    ]
    np.testing.assert_array_almost_equal(sim.toarray(), expected_similarity)


def test_compute_session_weights(algo, mini_training_dataset, mini_test_dataset):
    algo.fit(mini_training_dataset)

    sim = algo._compute_session_similarity(mini_test_dataset.last_timestamps_matrix)

    session_weights = algo._compute_session_similarity_weights(mini_test_dataset.last_timestamps_matrix, sim)

    expected_weights = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [np.exp(-2 / 5), np.exp(-3 / 5), 0, 0],
        [np.exp(-4 / 5), np.exp(-5 / 5), 0, 0],
    ]

    np.testing.assert_array_almost_equal(session_weights.toarray(), expected_weights)


def test_compute_prediction_scores(algo, mini_training_dataset, mini_test_dataset):
    algo.fit(mini_training_dataset)

    # fmt: off
    session_similarities = csr_matrix(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0]
        ]
    )
    # fmt: on
    predictions = algo._compute_prediction_scores(session_similarities, mini_test_dataset)

    # fmt: off
    expected_predictions = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [np.exp(-1), 0, np.exp(-1) + np.exp(-1), 0],
        [np.exp(-2), np.exp(-1), np.exp(-1), 0]
    ]
    # fmt: on

    np.testing.assert_array_equal(predictions.toarray(), expected_predictions)


def test_compute_prediction_scores_single_neighbour(algo, mini_training_dataset, mini_test_dataset):
    algo.fit(mini_training_dataset)

    # fmt: off
    session_similarities = csr_matrix(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
    )
    # fmt: on
    predictions = algo._compute_prediction_scores(session_similarities, mini_test_dataset)

    # fmt: off
    expected_predictions = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [np.exp(-1), 0, np.exp(-1), 0],
        [0, 0, np.exp(-1), 0]
    ]
    # fmt: on

    np.testing.assert_array_equal(predictions.toarray(), expected_predictions)


def test_predict(algo, mini_training_dataset, mini_test_dataset):
    algo.fit(mini_training_dataset)

    predictions = algo.predict(mini_test_dataset)

    sim_u_2_i_0 = np.exp(-12 / 5) / np.sqrt(6)
    sim_u_2_i_1 = 0
    sim_u_2_i_2 = np.exp(-12 / 5) / np.sqrt(6) + np.exp(-8 / 5) / 2
    sim_u_2_i_3 = 0

    sim_u_3_i_0 = np.exp(-19 / 5) / np.sqrt(6)
    sim_u_3_i_1 = np.exp(-14 / 5) / np.sqrt(6)
    sim_u_3_i_2 = (np.exp(-2) + np.exp(-3)) / 2
    sim_u_3_i_3 = 0

    # fmt: off
    expected_predictions = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [sim_u_2_i_0, sim_u_2_i_1, sim_u_2_i_2, sim_u_2_i_3],
        [sim_u_3_i_0, sim_u_3_i_1, sim_u_3_i_2, sim_u_3_i_3],
    ]
    # fmt: on

    np.testing.assert_array_almost_equal(predictions.toarray(), expected_predictions)


def test_fit_no_interaction_matrix(algo, mat):
    with pytest.raises(TypeError):
        algo.fit(mat.binary_values)


def test_fit_no_timestamps(algo, mat):
    with pytest.raises(ValueError):
        algo.fit(mat.eliminate_timestamps())


def test_predict_no_interaction_matrix(algo, mat):
    algo.fit(mat)
    with pytest.raises(TypeError):
        algo.predict(mat.binary_values)


def test_predict_no_timestamps(algo, mat):
    algo.fit(mat)
    with pytest.raises(ValueError):
        algo.predict(mat.eliminate_timestamps())
