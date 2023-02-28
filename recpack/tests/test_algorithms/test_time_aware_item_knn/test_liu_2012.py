# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from math import log
import numpy as np
import pytest

from recpack.algorithms import TARSItemKNNLiu2012
from recpack.algorithms.time_aware_item_knn.liu_2012 import LiuDecay


def test_liu_decay_order_preservation():
    for i in range(100):
        a = np.random.rand()
        b = np.random.rand()

        decay = np.random.randint(2, 10)
        output = LiuDecay(decay)(np.array([a, b]))
        # Events further from the first interaction get a higher weight
        assert (output[0] < output[1]) == (a < b)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([0.1, 0.3]), 2, np.array([log(0.1 + 1, 2) + 1, log(0.3 + 1, 2) + 1])),
        (np.array([0.1, 0.3]), 3, np.array([log(2 * 0.1 + 1, 3) + 1, log((2 * 0.3) + 1, 3) + 1])),
    ],
)
def test_liu_decay(input, decay, expected_output):
    result = LiuDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize("decay", [-1, 0, 1])
def test_liu_decay_bad_input(decay):
    with pytest.raises(ValueError):
        LiuDecay.validate_decay(decay)


def test_compute_users_first_interaction(mat_no_zero_timestamp):
    algorithm = TARSItemKNNLiu2012()

    first_interactions = algorithm._compute_users_first_interaction(mat_no_zero_timestamp)

    expected_result = np.array([[3, 2, 1, 3, 1, 3]]).T

    np.testing.assert_array_equal(first_interactions, expected_result)


def test_add_decay_to_interaction_matrix(mat_no_zero_timestamp):
    algorithm = TARSItemKNNLiu2012()

    weighted_mat = algorithm._add_decay_to_fit_matrix(mat_no_zero_timestamp)
    expected_result = np.array(
        [
            [log(1 / 4 + 1, 2) + 1, log(0 + 1, 2) + 1, 0, 0, 0],
            [0, 0, log(1, 2) + 1, log(3 / 5 + 1, 2) + 1, 0],
            [log(1, 2) + 1, log(1 / 2 + 1, 2) + 1, 0, 0, 0],
            [0, 0, log(1, 2) + 1, 0, log(2 / 5 + 1, 2) + 1],
            [log(1, 2) + 1, log(1 / 2 + 1, 2) + 1, 0, 0, 0],
            [0, 0, log(1, 2) + 1, 0, 0],
        ]
    )

    np.testing.assert_array_equal(weighted_mat.toarray(), expected_result)
