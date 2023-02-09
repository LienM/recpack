# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import scipy.sparse
import numpy as np
from unittest.mock import MagicMock

from recpack.algorithms.experimental import EASE_XY


@pytest.fixture()
def data():
    """
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [1 0 1],
            [1 0 1],
            [1 1 1],
            [1 0 1]
        ]

    """

    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))
    return data


@pytest.fixture()
def data_y():
    """
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [0 0 1],
            [0 0 1],
            [0 1 1],
            [0 0 1]
        ]

    """

    values = [1] * 5
    users = [0, 1, 2, 2, 3]
    items = [2, 2, 1, 2, 2]
    d = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))
    return d


def test_XY_same_data(data):
    algo = EASE_XY(l2=0.03)

    algo.fit(data, data)

    # Make sure the predictions "make sense"
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    np.testing.assert_almost_equal(result[2, 0], 1, decimal=1)
    np.testing.assert_almost_equal(result[0, 2], 1, decimal=1)


def test_XY(data, data_y):
    algo = EASE_XY(l2=0.03)

    algo.fit(data, data_y)

    # Make sure the predictions "make sense"
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    np.testing.assert_almost_equal(result[2, 0], 0, decimal=1)
    np.testing.assert_almost_equal(result[0, 2], 1, decimal=1)


def test_XY_density(data):
    algo = EASE_XY(l2=0.03, density=0.2)

    # Mocking the prune functionality, so we can assert it was called.
    algo._prune = MagicMock()

    algo.fit(data, data)

    algo._prune.assert_called_once()


def test_XY_alpha(data):
    # We will compare the scores for different alphas
    #  to check they are correctly applied
    algo_1 = EASE_XY(l2=0.03, alpha=1)
    algo_2 = EASE_XY(l2=0.03, alpha=0)
    algo_3 = EASE_XY(l2=0.03, alpha=2)

    algo_1.fit(data, data)
    algo_2.fit(data, data)
    algo_3.fit(data, data)

    # c(0) = 4
    # => similarity score with item 0 gets divided by c(0) ** alpha
    np.testing.assert_almost_equal(
        algo_1.similarity_matrix_[1, 0], algo_2.similarity_matrix_[1, 0] / 4
    )
    np.testing.assert_almost_equal(
        algo_1.similarity_matrix_[2, 1], algo_2.similarity_matrix_[2, 1]
    )
    np.testing.assert_almost_equal(
        algo_1.similarity_matrix_[1, 0] / 4, algo_3.similarity_matrix_[1, 0]
    )
