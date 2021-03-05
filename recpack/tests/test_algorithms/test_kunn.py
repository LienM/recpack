from math import sqrt

import numpy
from scipy.sparse import csr_matrix

from recpack.algorithms import KUNN, ItemKNN


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
    expected_Cu_rooted = [[1.0 / sqrt(7), 1.0 / sqrt(5), 1.0 / sqrt(10)]]
    expected_Ci_rooted = [[1.0 / sqrt(7), 1.0 / sqrt(6), 1.0 / sqrt(9)]]

    numpy.testing.assert_almost_equal(Xscaled.todense(), expected_Xscaled)
    numpy.testing.assert_almost_equal(Cu_rooted, expected_Cu_rooted)
    numpy.testing.assert_almost_equal(Ci_rooted, expected_Ci_rooted)


def test_kunn_fit():
    kunn = KUNN(Ku=1, Ki=1)

    values = [1, 1, 1, 1, 1, 1]
    users = [0, 1, 1, 2, 2, 2]
    items = [2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)))

    kunn.fit(test_matrix)

    knni_values = [(sqrt(2) + sqrt(3)) / 6, 1 / sqrt(6), (sqrt(2) + sqrt(3)) / 6]
    knni_items_x = [0, 1, 2]
    knni_items_y = [2, 0, 0]
    knni_true = csr_matrix((knni_values, (knni_items_x, knni_items_y)))

    numpy.testing.assert_almost_equal(knni_true.todense(), kunn.knn_i_.todense())


def test_kunn_predict():
    kunn = KUNN(Ku=1, Ki=1)

    values = [1, 1, 1, 1, 1, 1, 1]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)), shape=(5, 3))

    kunn.fit(test_matrix)

    # Test the prediction
    values_pred = [1, 1, 1, 1]
    users_pred = [3, 3, 4, 4]
    items_pred = [0, 1, 1, 2]
    pred_matrix = csr_matrix((values_pred, (users_pred, items_pred)), shape=test_matrix.shape)
    prediction = kunn.predict(pred_matrix)

    pred_true_values = [(122/815 + 4*sqrt(5209)/815)**2, (122/815 + 4*sqrt(5209)/815)**2,
                        sqrt(3*sqrt(401)/1478 + 931/1478),
                        (-83/244 + sqrt(75331)/244)**2, (-83/244 + sqrt(75331)/244)**2]
    pred_true_users = [3, 3, 3, 4, 4]
    pred_true_items = [0, 1, 2, 1, 2]
    pred_true = csr_matrix((pred_true_values, (pred_true_users, pred_true_items)), shape=prediction.shape)

    numpy.testing.assert_almost_equal(prediction.todense(), pred_true.todense())
