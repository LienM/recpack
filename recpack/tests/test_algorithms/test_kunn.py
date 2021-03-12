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


def test_predict_k_1():
    kunn = KUNN(Ku=1, Ki=1)

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

    prediction = kunn.predict(pred_matrix)

    itemknn = kunn._fit_item_knn(test_matrix)
    userknn = kunn._fit_user_knn(pred_matrix)

    # Manual computation of the formulas in the paper
    # We'll compute similarity 3, 2
    u = 3
    i = 2

    ## USER SIMILARITY ##

    # use ItemKNN class to compute user neighbours
    # By fitting it on the transpose of the combination of the test and pred matrices
    # In this case test_matrix and pred matrix are fully disjunct
    # TODO: add a test with non fully disjunct test and pred matrices,
    #       To make sure it can be used with the other splitters as well.

    # Get the one user similiar to user u
    v = numpy.argmax(userknn[u])
    # We are "lucky", the most similar user is one from the training users,
    # so don't need to do finicking to remove the unwanted similarities
    assert v == 2

    # User 2 has seen item 2 we are trying to predict
    # -> the R_v_i term in the formula is 1
    # compute the c(v) value
    c_v = (pred_matrix + test_matrix)[v].nnz
    c_u = (pred_matrix + test_matrix)[u].nnz

    # Second summation, over all items v and u have in common
    # user 3 and user 2 have 2 items in common (0 and 1)
    c_j_0 = (pred_matrix + test_matrix)[:, 0].nnz
    c_j_1 = (pred_matrix + test_matrix)[:, 1].nnz

    # Compute user similarity.
    # 1/sqrt(cv) + sum(j in [0, 1] 1/sqrt(c(j)))
    s_u = (1 / c_v ** 0.5) * ((1 / c_j_0 ** 0.5) + (1 / c_j_1 ** 0.5))

    ## ITEM SIMILARITY ##

    # K = 1 -> argmax gives us the most similar item
    j = numpy.argmax(itemknn[i])
    assert j == 0

    # c(j) in paper, only need it for the one j
    # User u has interacted with item 0 -> R_u_j = 1
    c_j = test_matrix[:, j].nnz
    c_i = test_matrix[:, i].nnz

    # 2 users have seen both items 0 and 2 => user 1,2
    numpy.testing.assert_array_equal(
        test_matrix[:, j].multiply(test_matrix[:, i]).toarray().nonzero()[0],
        numpy.array([1, 2]),
    )

    # Compute history lengths of the two users
    c_v_0 = test_matrix[1].nnz
    c_v_2 = test_matrix[2].nnz

    # Compute item similarity
    # 1/sqrt(c_j) * sum(v in [1,2]) 1/sqrt(c(v))
    s_i = (1 / c_j ** 0.5) * ((1 / c_v_0 ** 0.5) + (1 / c_v_2) ** 0.5)

    ## FINAL SCORE ##
    # (s_u + s_i) / (sqrt(c(u)*c(i)))
    s_u_i = (s_u + s_i) / ((c_u * c_i) ** 0.5)

    numpy.testing.assert_almost_equal(prediction[3, 2], s_u_i)


def test_predict_k_2():
    kunn = KUNN(Ku=2, Ki=2)

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

    prediction = kunn.predict(pred_matrix)

    # Fit the KNN models
    itemknn = kunn._fit_item_knn(test_matrix)
    userknn = kunn._fit_user_knn(pred_matrix)

    # Predict score for user 3, item 2
    u = 3
    i = 2

    # V is set of users that are neighbours of u
    V = userknn[u].nonzero()[1]
    # We are 'lucky' our new users are not similar to the other new users
    #     -> No leakage of info
    numpy.testing.assert_array_equal(numpy.sort(V), numpy.array([1, 2]))

    # J is set of items that are neighbours of i
    J = itemknn[i].nonzero()[1]

    # Compute 1/sqrt(c(v)) for each v
    one_over_sqrt_v = {v: 1 / test_matrix[v].nnz ** 0.5 for v in V}
    # Compute 1/sqrt(c(j)) for each j
    one_over_sqrt_j = {j: 1 / test_matrix[j].nnz ** 0.5 for j in J}

    ## USER SIMILARITY ##
    # Iteratively compute the user sim
    score = 0
    for v in V:  # 1st sum : v in KNN(u)
        if (pred_matrix + test_matrix)[v, i] != 0:  # R_v_i in the first sum.
            # Compute transitive part
            trans_sum = 0
            for j in (pred_matrix + test_matrix)[u].nonzero()[
                1
            ]:  # Second sum, with R_u_i = 1

                if (pred_matrix + test_matrix)[v, j] != 0:  # R_u_j = 1 clause in sum

                    c_j = (pred_matrix + test_matrix)[:, j].nnz

                    # "HACK" To count the interaction of user u in the pred matrix
                    if (pred_matrix + test_matrix)[u, j] == 0:
                        c_j += 1
                    # End of "HACK"

                    trans_sum += 1 / c_j ** 0.5

            score += trans_sum * one_over_sqrt_v[v]

    user_sim = score

    ## ITEM SIMILARITY ##
    # Compute the item similarity iteratively
    score = 0
    for j in J:  # 1st sum j in KNN(i)
        if pred_matrix[u, j] != 0:  # R_u_j value
            trans_sum = 0
            for v in test_matrix[:, j].nonzero()[0]:  # Second sum, with R_v_j clause
                if test_matrix[v, i] != 0:  # R_v_i clause in second sum
                    c_v = test_matrix[v, :].nnz
                    trans_sum += 1 / c_v ** 0.5

            score += trans_sum * one_over_sqrt_j[j]

    item_sim = score

    ## FINAL SCORE ##
    final_score = (user_sim + item_sim) / (
        (pred_matrix + test_matrix)[u].nnz * test_matrix[:, i].nnz
    ) ** 0.5

    numpy.testing.assert_almost_equal(prediction[u, i], final_score)
