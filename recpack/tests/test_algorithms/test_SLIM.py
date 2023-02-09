# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms import SLIM


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
    data = csr_matrix((values, (users, items)), shape=(5, 3))
    return data


@pytest.fixture()
def data_negatives():
    """
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [1 1 0 0],
            [1 1 0 0],
            [0 0 1 1],
            [0 0 1 1]
        ]

    """

    values = [1] * 8
    users = [0, 0, 1, 1, 2, 2, 3, 3]
    items = [0, 1, 0, 1, 2, 3, 2, 3]
    data = csr_matrix((values, (users, items)), shape=(4, 4))
    return data


def test_slim(data):
    algo = SLIM()

    algo.fit(data)

    # Make sure the predictions "make sense"
    _in = csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    # Item 0 is closer to item 2
    assert result[2, 0] > result[2, 1]
    # Zero diagonal
    np.testing.assert_array_almost_equal(result[[0, 1, 2], [0, 1, 2]], 0)


def test_slim_negatives(data_negatives):
    algo = SLIM()

    algo.fit(data_negatives)

    # Make sure the predictions "make sense"
    _in = csr_matrix(([1, 1, 1, 1], ([0, 1, 2, 3], [0, 1, 2, 3])), shape=(4, 4))
    result = algo.predict(_in)

    # No similarity between item 0 and 2
    assert result[2, 0] == 0

    algo2 = SLIM(ignore_neg_weights=False)
    algo2.fit(data_negatives)

    # Make sure the predictions "make sense"
    result2 = algo2.predict(_in)

    # Item 0 is closer to item 2
    assert result2[2, 0] < 0
