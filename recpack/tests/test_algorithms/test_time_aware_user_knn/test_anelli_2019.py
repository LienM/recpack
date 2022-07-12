import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from recpack.algorithms.time_aware_user_knn.anelli_2019 import TARSUserKNNAnelli


@pytest.fixture()
def algorithm():
    return TARSUserKNNAnelli(decay=1 / 2, min_number_of_recommendations=2)


@pytest.fixture()
def algorithm_at_least_4_recos():
    return TARSUserKNNAnelli(decay=1 / 2, min_number_of_recommendations=4)


def test_fit(algorithm, mat_no_zero_timestamp):
    print(mat_no_zero_timestamp.last_timestamps_matrix.toarray())
    algorithm.fit(mat_no_zero_timestamp)
    assert algorithm.precursors_.shape == (mat_no_zero_timestamp.shape[0], mat_no_zero_timestamp.shape[0])

    expected_value = np.array(
        [
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )

    np.testing.assert_array_equal(algorithm.precursors_.toarray(), expected_value)


def test_predict(algorithm, mat_no_zero_timestamp):
    algorithm.fit(mat_no_zero_timestamp)

    pred = algorithm.predict(mat_no_zero_timestamp)

    expected_prediction = np.array(
        [
            [2 * np.exp(-3 / 2), 2 * np.exp(-2), 0, 0, 0],
            [3, 3, 3, 1, 1],
            [3, 3, 3, 1, 1],
            [0, 0, np.exp(-1), np.exp(-1 / 2), 0],
            [3, 3, 3, 1, 1],
            [0, 0, np.exp(-1), np.exp(-1 / 2), 0],
        ]
    )

    np.testing.assert_array_almost_equal(pred.toarray(), expected_prediction)


def test_predict_w_fallback(algorithm_at_least_4_recos, mat_no_zero_timestamp):
    algorithm_at_least_4_recos.fit(mat_no_zero_timestamp)

    pred = algorithm_at_least_4_recos.predict(mat_no_zero_timestamp)
    print(pred.toarray())
    expected_prediction = np.array(
        [
            [2 * np.exp(-3 / 2), 2 * np.exp(-2), (2 * np.exp(-2) / 4) * 3, 2 * np.exp(-2) / 4, 2 * np.exp(-2) / 4],
            [3, 3, 3, 1, 1],
            [3, 3, 3, 1, 1],
            [(np.exp(-1) / 4) * 3, (np.exp(-1) / 4) * 3, np.exp(-1), np.exp(-1 / 2), (np.exp(-1) / 4) * 1],
            [3, 3, 3, 1, 1],
            [(np.exp(-1) / 4) * 3, (np.exp(-1) / 4) * 3, np.exp(-1), np.exp(-1 / 2), (np.exp(-1) / 4) * 1],
        ]
    )

    np.testing.assert_array_almost_equal(pred.toarray(), expected_prediction)
