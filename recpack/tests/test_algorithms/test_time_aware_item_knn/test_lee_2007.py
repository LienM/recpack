import numpy as np

from recpack.algorithms import TARSItemKNNLee, TARSItemKNNLee_W3, TARSItemKNNLee_W5


def test_weight_matrix_W5():
    # fmt: off
    expected_matrix = np.array([
        [0.2, 1.2, 2.2, 3.2, 4.2],
        [0.4, 1.4, 2.4, 3.4, 4.4],
        [0.6, 1.6, 2.6, 3.6, 4.6],
        [0.8, 1.8, 2.8, 3.8, 4.8],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ])
    # fmt: on

    algorithm = TARSItemKNNLee_W5()
    np.testing.assert_array_almost_equal(algorithm.weight_matrix, expected_matrix)


def test_weight_matrix_W4():
    # fmt: off
    expected_matrix = np.array([
        [0.25, 1.25, 2.25, 3.25],
        [0.50, 1.50, 2.50, 3.50],
        [0.75, 1.75, 2.75, 3.75],
        [1.00, 2.00, 3.00, 4.00],
    ])
    # fmt: on

    algorithm = TARSItemKNNLee(W=4)
    np.testing.assert_array_almost_equal(algorithm.weight_matrix, expected_matrix)


def test_compute_launch_times(mat):
    algorithm = TARSItemKNNLee()

    launch_times = algorithm._compute_launch_times(mat)
    assert launch_times.shape == (mat.shape[1],)
    expected_launch_times = np.array([0, 1, 1, 4, 4])
    np.testing.assert_array_equal(launch_times, expected_launch_times)


def test_add_decay_to_interaction_matrix_W3(mat):
    print(mat.last_timestamps_matrix.toarray())
    algorithm = TARSItemKNNLee_W3()
    expected_matrix = np.array(
        [
            [1.7, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 3.3, 0.0],
            [0.0, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0, 3.3],
            [0.0, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0, 0.0],
        ]
    )

    weighted_matrix = algorithm._add_decay_to_interaction_matrix(mat).toarray()

    np.testing.assert_array_equal(weighted_matrix, expected_matrix)


def test_add_decay_to_interaction_matrix_W4(mat):
    print(mat.last_timestamps_matrix.toarray())
    algorithm = TARSItemKNNLee(W=4)
    expected_matrix = np.array(
        [
            [2.25, 1.25, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.25, 4.00, 0.00],
            [0.00, 0.25, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.25, 0.00, 4.00],
            [0.00, 0.25, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.25, 0.00, 0.00],
        ]
    )

    weighted_matrix = algorithm._add_decay_to_interaction_matrix(mat).toarray()

    np.testing.assert_array_equal(weighted_matrix, expected_matrix)


def test_fit(mat):
    algorithm = TARSItemKNNLee_W5()
    algorithm.fit(mat)

    assert algorithm.similarity_matrix_.shape == (mat.shape[1], mat.shape[1])


def test_predict(mat):
    algorithm = TARSItemKNNLee_W5()
    algorithm.fit(mat)

    predictions = algorithm.predict(mat)
    assert predictions.shape == mat.shape
