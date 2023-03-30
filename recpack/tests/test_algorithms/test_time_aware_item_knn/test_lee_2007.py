# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np

from recpack.algorithms import TARSItemKNNLee

def test_compute_launch_times(mat):
    algorithm = TARSItemKNNLee()

    launch_times = algorithm._compute_launch_times(mat)
    assert launch_times.shape == (mat.shape[1],)
    expected_launch_times = np.array([0, 1, 1, 4, 4])
    np.testing.assert_array_equal(launch_times, expected_launch_times)


def test_add_decay_to_fit_matrix_W3(mat):
    algorithm = TARSItemKNNLee(w=3)
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

    weighted_matrix = algorithm._add_decay_to_fit_matrix(mat).toarray()

    np.testing.assert_array_equal(weighted_matrix, expected_matrix)


def test_add_decay_to_fit_matrix_W5(mat):
    algorithm = TARSItemKNNLee(w=5)
    expected_matrix = np.array(
        [
            [2.20, 1.20, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.20, 3.80, 0.00],
            [0.00, 0.2, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.2, 0.00, 3.80],
            [0.00, 0.2, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.2, 0.00, 0.00],
        ]
    )

    weighted_matrix = algorithm._add_decay_to_fit_matrix(mat).toarray()

    np.testing.assert_array_equal(weighted_matrix, expected_matrix)


def test_fit(mat):
    algorithm = TARSItemKNNLee()
    algorithm.fit(mat)

    assert algorithm.similarity_matrix_.shape == (mat.shape[1], mat.shape[1])


def test_predict(mat):
    algorithm = TARSItemKNNLee(w=5)
    algorithm.fit(mat)

    predictions = algorithm.predict(mat)
    assert predictions.shape == mat.shape
