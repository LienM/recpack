# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

from recpack.algorithms.time_aware_item_knn import TARSItemKNNLiu


@pytest.fixture()
def algorithm() -> TARSItemKNNLiu:
    return TARSItemKNNLiu(K=2, fit_decay=0.5, predict_decay=0.5)


def test_check_input(algorithm, matrix_sessions):
    # No error when checking type
    algorithm._transform_fit_input(matrix_sessions)
    algorithm._transform_predict_input(matrix_sessions)

    with pytest.raises(TypeError) as type_error:
        algorithm._transform_fit_input(matrix_sessions.binary_values)

    assert type_error.match(
        "TARSItemKNNLiu requires Interaction Matrix as input. Got <class 'scipy.sparse._csr.csr_matrix'>."
    )

    with pytest.raises(TypeError) as type_error:
        algorithm._transform_predict_input(matrix_sessions.binary_values)

    assert type_error.match(
        "TARSItemKNNLiu requires Interaction Matrix as input. Got <class 'scipy.sparse._csr.csr_matrix'>."
    )

    with pytest.raises(ValueError) as value_error:
        algorithm._transform_fit_input(matrix_sessions.eliminate_timestamps())

    assert value_error.match("TARSItemKNNLiu requires timestamp information in the InteractionMatrix.")

    with pytest.raises(ValueError) as value_error:
        algorithm._transform_predict_input(matrix_sessions.eliminate_timestamps())

    assert value_error.match("TARSItemKNNLiu requires timestamp information in the InteractionMatrix.")


def test_fit(algorithm, mat):
    algorithm.fit(mat)

    assert algorithm.similarity_matrix_.shape == (mat.shape[1], mat.shape[1])

    # TODO: value checks, we know our weighting func works, so we'd only check that values are passed correctly


def test_predict(algorithm, mat):
    algorithm.fit(mat)
    predictions = algorithm.predict(mat)

    assert mat.shape == predictions.shape
    # TODO: value check?
