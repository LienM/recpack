# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pytest

from recpack.algorithms.time_aware_item_knn import TARSItemKNN, TARSItemKNNCoocDistance


@pytest.fixture(params=["cosine", "conditional_probability", "pearson"])
def algorithm(request) -> TARSItemKNN:
    return TARSItemKNN(K=2, fit_decay=0.5, predict_decay=0.5, similarity=request.param)


@pytest.fixture()
def cooc_algorithm():
    return TARSItemKNNCoocDistance(K=2, fit_decay=0.5, similarity="cooc")


def test_check_input(algorithm, matrix_sessions):
    # No error when checking type
    algorithm._transform_fit_input(matrix_sessions)
    algorithm._transform_predict_input(matrix_sessions)

    with pytest.raises(TypeError) as type_error:
        algorithm._transform_fit_input(matrix_sessions.binary_values)

    assert type_error.match(
        "TARSItemKNN requires Interaction Matrix as input. Got <class 'scipy.sparse._csr.csr_matrix'>."
    )

    with pytest.raises(TypeError) as type_error:
        algorithm._transform_predict_input(matrix_sessions.binary_values)

    assert type_error.match(
        "TARSItemKNN requires Interaction Matrix as input. Got <class 'scipy.sparse._csr.csr_matrix'>."
    )

    with pytest.raises(ValueError) as value_error:
        algorithm._transform_fit_input(matrix_sessions.eliminate_timestamps())

    assert value_error.match("TARSItemKNN requires timestamp information in the InteractionMatrix.")

    with pytest.raises(ValueError) as value_error:
        algorithm._transform_predict_input(matrix_sessions.eliminate_timestamps())

    assert value_error.match("TARSItemKNN requires timestamp information in the InteractionMatrix.")


def test_add_decay_to_fit_matrix(algorithm, mat):
    result = algorithm._add_decay_to_fit_matrix(mat)

    MAX_TS = mat.timestamps.max()
    NOW = MAX_TS + 1
    expected_result = np.array(
        [
            [np.exp(-(NOW - 3) / 2), np.exp(-(NOW - 2) / 2), 0, 0, 0],
            [0, 0, np.exp(-(NOW - 1) / 2), np.exp(-(NOW - 4) / 2), 0],
            [np.exp(-(NOW - 0) / 2), np.exp(-(NOW - 1) / 2), 0, 0, 0],
            [0, 0, np.exp(-(NOW - 2) / 2), 0, np.exp(-(NOW - 4) / 2)],
            [np.exp(-(NOW - 0) / 2), np.exp(-(NOW - 1) / 2), 0, 0, 0],
            [0, 0, np.exp(-(NOW - 2) / 2), 0, 0],
        ]
    )

    np.testing.assert_array_equal(result.toarray(), expected_result)


def test_add_decay_to_predict_matrix(algorithm, mat):
    result = algorithm._add_decay_to_predict_matrix(mat)

    MAX_TS = mat.timestamps.max()
    NOW = MAX_TS + 1
    expected_result = np.array(
        [
            [np.exp(-(NOW - 3) / 2), np.exp(-(NOW - 2) / 2), 0, 0, 0],
            [0, 0, np.exp(-(NOW - 1) / 2), np.exp(-(NOW - 4) / 2), 0],
            [np.exp(-(NOW - 0) / 2), np.exp(-(NOW - 1) / 2), 0, 0, 0],
            [0, 0, np.exp(-(NOW - 2) / 2), 0, np.exp(-(NOW - 4) / 2)],
            [np.exp(-(NOW - 0) / 2), np.exp(-(NOW - 1) / 2), 0, 0, 0],
            [0, 0, np.exp(-(NOW - 2) / 2), 0, 0],
        ]
    )

    np.testing.assert_array_equal(result.toarray(), expected_result)


def test_fit(algorithm, mat):
    algorithm.fit(mat)

    assert algorithm.similarity_matrix_.shape == (mat.shape[1], mat.shape[1])

    # TODO: value checks, we know our weighting func works, so we'd only check that values are passed correctly


def test_predict(algorithm, mat):
    algorithm.fit(mat)
    predictions = algorithm.predict(mat)

    assert mat.shape == predictions.shape
    # TODO: value check?


@pytest.mark.parametrize(
    "similarity, expected_similarities",
    [
        (
            "cooc",
            np.array(
                [
                    [0, np.exp(-1 / 2) * 3, 0, 0, 0],
                    [np.exp(-1 / 2) * 3, 0, 0, 0, 0],
                    [0, 0, 0, np.exp(-3 / 2), np.exp(-2 / 2)],
                    [0, 0, np.exp(-3 / 2), 0, 0],
                    [0, 0, np.exp(-2 / 2), 0, 0],
                ]
            ),
        ),
        (
            "conditional_probability",
            np.array(
                [
                    [0, np.exp(-1 / 2), 0, 0, 0],
                    [np.exp(-1 / 2), 0, 0, 0, 0],
                    [0, 0, 0, np.exp(-3 / 2) / 3, np.exp(-2 / 2) / 3],
                    [0, 0, np.exp(-3 / 2), 0, 0],
                    [0, 0, np.exp(-2 / 2), 0, 0],
                ]
            ),
        ),
    ],
)
def test_cooc_decay_fit(similarity, expected_similarities, mat_no_zero_timestamp):
    algorithm = TARSItemKNNCoocDistance(K=2, fit_decay=0.5, similarity=similarity)
    algorithm.fit(mat_no_zero_timestamp)

    np.testing.assert_array_almost_equal(algorithm.similarity_matrix_.toarray(), expected_similarities)
