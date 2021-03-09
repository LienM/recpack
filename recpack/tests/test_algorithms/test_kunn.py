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
    expected_Xscaled = [
        [
            0.0,
            2 * (1.0 / sqrt(7)) * (1.0 / sqrt(6)),
            5 * (1.0 / sqrt(7)) * (1.0 / sqrt(9)),
        ],
        [
            4 * (1.0 / sqrt(5)) * (1.0 / sqrt(7)),
            0.0,
            1 * (1.0 / sqrt(5)) * (1.0 / sqrt(9)),
        ],
        [
            3 * (1.0 / sqrt(10)) * (1.0 / sqrt(7)),
            4 * (1.0 / sqrt(10)) * (1.0 / sqrt(6)),
            3 * (1.0 / sqrt(10)) * (1.0 / sqrt(9)),
        ],
    ]
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
    pred_matrix = csr_matrix(
        (values_pred, (users_pred, items_pred)), shape=test_matrix.shape
    )
    prediction = kunn.predict(pred_matrix)

    pred_true_values = [
        (122 / 815 + 4 * sqrt(5209) / 815) ** 2,
        (122 / 815 + 4 * sqrt(5209) / 815) ** 2,
        sqrt(3 * sqrt(401) / 1478 + 931 / 1478),
        (-83 / 244 + sqrt(75331) / 244) ** 2,
        (-83 / 244 + sqrt(75331) / 244) ** 2,
    ]
    pred_true_users = [3, 3, 3, 4, 4]
    pred_true_items = [0, 1, 2, 1, 2]
    pred_true = csr_matrix(
        (pred_true_values, (pred_true_users, pred_true_items)), shape=prediction.shape
    )

    numpy.testing.assert_almost_equal(prediction.todense(), pred_true.todense())


def test_kunn_item_knn():
    values = [1, 1, 1, 1, 1, 1, 1]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)), shape=(5, 3))

    Ki = 2
    Ku = 2
    kunn = KUNN(Ku=Ku, Ki=Ki)
    item_knn = kunn._fit_item_knn(test_matrix)

    assert item_knn.shape == (test_matrix.shape[1],) * 2
    for i in range(test_matrix.shape[1]):
        assert item_knn[i, :].nnz <= Ki

    iknn_values = [
        1 / (2 * sqrt(3)),
        (sqrt(2) + sqrt(3)) / 6,
        1 / (2 * sqrt(3)),
        (sqrt(2) + sqrt(3)) / 6,
        (sqrt(2) + sqrt(3)) / 6,
        (sqrt(2) + sqrt(3)) / 6,
    ]
    iknn_item_1 = [0, 0, 1, 1, 2, 2]
    iknn_item_2 = [1, 2, 0, 2, 0, 1]
    pred_iknn = csr_matrix(
        (iknn_values, (iknn_item_1, iknn_item_2)), shape=(test_matrix.shape[1],) * 2
    )

    numpy.testing.assert_almost_equal(item_knn.todense(), pred_iknn.todense())


def test_kunn_user_knn():
    Ki = 2
    Ku = 2
    kunn = KUNN(Ku=Ku, Ki=Ki)

    values = [1, 1, 1, 1, 1, 1, 1]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)), shape=(5, 3))

    kunn.fit(test_matrix)

    values_pred = [1, 1, 1, 1]
    users_pred = [3, 3, 4, 4]
    items_pred = [0, 1, 1, 2]
    pred_matrix = csr_matrix(
        (values_pred, (users_pred, items_pred)), shape=test_matrix.shape
    )

    user_knn = kunn._fit_user_knn(pred_matrix)
    assert user_knn.shape == (test_matrix.shape[0],) * 2
    for u in range(test_matrix.shape[0]):
        assert user_knn[u, :].nnz <= Ku

    uknn_values = [
        1 / 4,
        1 / sqrt(6),
        1 / 4,
        (2 + sqrt(3)) / (6 * sqrt(2)),
        1 / sqrt(6),
        (2 + sqrt(3)) / (6 * sqrt(2)),
        1 / (2 * sqrt(3)),
        (2 + sqrt(3)) / (6 * sqrt(2)),
        1 / 2,
        1 / sqrt(6),
    ]
    uknn_user_1 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    uknn_user_2 = [1, 2, 0, 2, 0, 1, 1, 2, 0, 2]
    pred_uknn = csr_matrix(
        (uknn_values, (uknn_user_1, uknn_user_2)), shape=(test_matrix.shape[0],) * 2
    )
    numpy.testing.assert_almost_equal(user_knn.todense(), pred_uknn.todense())


def test_combination_csr_matrices():
    a = csr_matrix([[1, 1, 0], [0, 1, 1]])
    b = csr_matrix([[0, 1, 0], [1, 1, 0]])

    # a = csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(2, 3))
    # b = csr_matrix([[0, 1, 0], [1, 1, 1]])

    kunn = KUNN()
    combined = kunn._union_csr_matrices(a, b)

    result = csr_matrix([[1, 1, 0], [1, 1, 1]])
    assert a.shape == b.shape == combined.shape
    numpy.testing.assert_almost_equal(combined.todense(), result.todense())
