import numpy as np
import pytest

from recpack.algorithms.tars_itemknn import TARSItemKNN, TARSItemKNNLiu


@pytest.fixture
def algorithm_liu() -> TARSItemKNNLiu:
    return TARSItemKNNLiu(K=2, fit_decay=0.5, predict_decay=0.5)


@pytest.fixture(params=["cosine", "conditional_probability"])
def algorithm(request) -> TARSItemKNN:
    return TARSItemKNN(K=2, fit_decay=0.5, predict_decay=0.5, similarity=request.param)


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


def test_add_decay_to_interaction_matrix(algorithm, mat):
    result = algorithm._add_decay_to_interaction_matrix(mat, algorithm.fit_decay)

    MAX_TS = mat.timestamps.max()
    expected_result = np.array(
        [
            [np.exp(-(MAX_TS - 3) / 2), np.exp(-(MAX_TS - 2) / 2), 0, 0, 0],
            [0, 0, np.exp(-(MAX_TS - 1) / 2), np.exp(-(MAX_TS - 4) / 2), 0],
            [np.exp(-(MAX_TS - 0) / 2), np.exp(-(MAX_TS - 1) / 2), 0, 0, 0],
            [0, 0, np.exp(-(MAX_TS - 2) / 2), 0, np.exp(-(MAX_TS - 4) / 2)],
            [np.exp(-(MAX_TS - 0) / 2), np.exp(-(MAX_TS - 1) / 2), 0, 0, 0],
            [0, 0, np.exp(-(MAX_TS - 2) / 2), 0, 0],
        ]
    )

    np.testing.assert_array_equal(result.toarray(), expected_result)
