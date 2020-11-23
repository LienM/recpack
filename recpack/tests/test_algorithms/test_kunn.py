from math import sqrt

import numpy
from scipy.sparse import csr_matrix

from recpack.algorithms import KUNN


def test_kunn_calculate_scaled_matrices():
    kunn = KUNN(Ku=3, Ki=3)

    values = [2, 5, 4, 1, 3, 4, 3]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)))

    Xscaled, Cu_rooted, Ci_rooted = kunn._calculate_scaled_matrices(test_matrix)
    expected_Xscaled = [[0.0, 2 * (1.0 / sqrt(7)) * (1.0 / sqrt(6)), 5 * (1.0 / sqrt(7)) * (1.0 / sqrt(9))],
                        [4 * (1.0 / sqrt(5)) * (1.0 / sqrt(7)), 0.0, 1 * (1.0 / sqrt(5)) * (1.0 / sqrt(9))],
                        [3 * (1.0 / sqrt(10)) * (1.0 / sqrt(7)), 4 * (1.0 / sqrt(10)) * (1.0 / sqrt(6)),  3 * (1.0 / sqrt(10)) * (1.0 / sqrt(9))]]
    expected_Cu_rooted = [[1.0 / sqrt(7), 0.0, 0.0],
                          [0.0, 1.0 / sqrt(5), 0.0],
                          [0.0, 0.0, 1.0 / sqrt(10)]]
    expected_Ci_rooted = [[1.0 / sqrt(7), 0.0, 0.0],
                          [0.0, 1.0 / sqrt(6), 0.0],
                          [0.0, 0.0, 1.0 / sqrt(9)]]

    numpy.testing.assert_almost_equal(Xscaled.todense(), expected_Xscaled)
    numpy.testing.assert_almost_equal(Cu_rooted.todense(), expected_Cu_rooted)
    numpy.testing.assert_almost_equal(Ci_rooted.todense(), expected_Ci_rooted)
