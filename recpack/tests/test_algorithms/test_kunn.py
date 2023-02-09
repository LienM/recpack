# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from math import sqrt

import numpy
from scipy.sparse import csr, csr_matrix

from recpack.algorithms import KUNN
from recpack.algorithms.util import union_csr_matrices


def test_kunn_fit():
    kunn = KUNN(Ku=1, Ki=1)

    test_matrix = csr_matrix([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    kunn.fit(test_matrix)

    knni_values = [
        (sqrt(2) + sqrt(3)) / 6,
        1 / sqrt(6),
        (sqrt(2) + sqrt(3)) / 6
    ]
    knni_items_x = [2, 0, 0]
    knni_items_y = [0, 1, 2]
    knni_true = csr_matrix((knni_values, (knni_items_x, knni_items_y)))

    numpy.testing.assert_almost_equal(
        knni_true.todense(), kunn.knn_i_.todense())


def test_kunn_item_knn():
    test_matrix = csr_matrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

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
        (iknn_values, (iknn_item_1, iknn_item_2)), shape=(
            test_matrix.shape[1],) * 2
    )

    numpy.testing.assert_almost_equal(item_knn.todense(), pred_iknn.todense())


def test_kunn_user_knn():
    Ki = 2
    Ku = 2
    kunn = KUNN(Ku=Ku, Ki=Ki)

    test_matrix = csr_matrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

    kunn.fit(test_matrix)

    # Construct prediction input matrix
    pred_matrix = csr_matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])

    user_knn = kunn._fit_user_knn(pred_matrix)
    assert user_knn.shape == (test_matrix.shape[0],) * 2
    for u in range(test_matrix.shape[0]):
        assert user_knn[u, :].nnz <= Ku

    # The expected similarities:
    # The goal of the user KNN comp is to find similar users
    # in the training dataset for each user in the predict dataset
    # So similarities from training users should be 0
    # We also don't want other prediction users to be used
    # for predicting items, that would leak data.

    # The expected values are computed below
    # Above each expected value the variables in the computation are written.
    # Counts of items c_i_X are written as the sum of occurrences in the
    #   training matrix + occurrences in the user's prediction history
    # Counts of users c_u_X are based on the training matrix for the neighbours
    #   And on the union of the prediction and training matrix for the centers.
    #   In this example both are disjunct, so overlap does not
    #   have to be taken into account

    # fmt: off
    uknn_values = [
        # sim(3, 0); c_u_3 = 2, c_u_0 = 2, c_i_1 = 2 + 1
        # NOT IN K NN
        # NOTE: we use argpartition(-K), since it keeps the original order of entries,
        # if there is a tie, the last user will get selected.
        # sim(3, 1); c_u_3 = 2, c_u_1 = 2, c_i_0 = 2 + 1
        1 / sqrt(2 * 2 * 3),
        # sim(3, 2); c_u_3 = 2, c_u_2 = 2, c_i_0 = 2 + 1, c_i_1 = 2 + 1
        2 / sqrt(2 * 3 * 3),
        # sim(4, 0); c_u_4 = 2, c_u_0 = 2, c_i_1 = 2 + 1, c_i_2 = 3 + 1
        (1 / sqrt(2 * 2 * 3)) + (1 / sqrt(2 * 2 * 4)),
        # sim(4, 1); c_u_4 = 2, c_u_1 = 2, c_i_2 = 3 + 1
        # NOT IN K NN
        # sim(4, 2); c_u_4 = 2, c_u_2 = 3, c_i_1 = 2 + 1, c_i_2 = 3 + 1
        (1 / sqrt(2 * 3 * 3)) + (1 / sqrt(2 * 3 * 4)),
    ]
    # fmt:on
    uknn_user_1 = [3, 3, 4, 4]
    uknn_user_2 = [1, 2, 0, 2]
    pred_uknn = csr_matrix(
        (uknn_values, (uknn_user_1, uknn_user_2)), shape=(
            test_matrix.shape[0],) * 2
    )
    numpy.testing.assert_almost_equal(user_knn.todense(), pred_uknn.todense())


def test_kunn_user_knn_full_overlap():
    Ki = 2
    Ku = 2
    kunn = KUNN(Ku=Ku, Ki=Ki)

    test_matrix = csr_matrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    kunn.fit(test_matrix)

    # Construct prediction input matrix as copy of train.
    pred_matrix = test_matrix.copy()

    user_knn = kunn._fit_user_knn(pred_matrix)
    assert user_knn.shape == (test_matrix.shape[0],) * 2
    for u in range(test_matrix.shape[0]):
        assert user_knn[u, :].nnz <= Ku

    # The expected similarities:

    uknn_values = [
        # sim(0, 1) & sim(1, 0)
        1 / sqrt(2 * 2 * 3),
        1 / sqrt(2 * 2 * 3),
        # sim(0, 2) & sim(2, 0)
        (1 / sqrt(2 * 3 * 3)) + (1 / sqrt(2 * 3 * 2)),
        (1 / sqrt(2 * 3 * 3)) + (1 / sqrt(2 * 3 * 2)),
        # sim(1, 2) & sim(2, 1)
        (1 / sqrt(2 * 3 * 3)) + (1 / sqrt(2 * 3 * 2)),
        (1 / sqrt(2 * 3 * 3)) + (1 / sqrt(2 * 3 * 2)),
    ]

    uknn_user_1 = [0, 1, 0, 2, 1, 2]
    uknn_user_2 = [1, 0, 2, 0, 2, 1]
    pred_uknn = csr_matrix(
        (uknn_values, (uknn_user_1, uknn_user_2)), shape=(
            test_matrix.shape[0],) * 2
    )
    numpy.testing.assert_almost_equal(user_knn.todense(), pred_uknn.todense())


def test_kunn_user_knn_partial_overlap():
    Ki = 2
    Ku = 2
    kunn = KUNN(Ku=Ku, Ki=Ki)

    test_matrix = csr_matrix([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])

    kunn.fit(test_matrix)

    pred_matrix = csr_matrix([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    user_knn = kunn._fit_user_knn(pred_matrix)
    assert user_knn.shape == (test_matrix.shape[0],) * 2
    for u in range(test_matrix.shape[0]):
        assert user_knn[u, :].nnz <= Ku

    # The expected similarities:
    uknn_values = [
        # sim(0, 1)
        1 / sqrt(2 * 2 * 2),
        # sim(0, 2)
        (1 / sqrt(2 * 2 * 2)),
        # sim(2, 0)
        (1 / sqrt(1 * 3 * 3)),
        # sim(2, 1)
        (1 / sqrt(2 * 3 * 3)) + (1 / sqrt(2 * 3 * 2)),
    ]

    uknn_user_1 = [0, 0, 2, 2]
    uknn_user_2 = [1, 2, 0, 1]
    pred_uknn = csr_matrix(
        (uknn_values, (uknn_user_1, uknn_user_2)), shape=(
            test_matrix.shape[0],) * 2
    )
    numpy.testing.assert_almost_equal(user_knn.todense(), pred_uknn.todense())


def test_combination_csr_matrices():
    a = csr_matrix([[1, 1, 0], [0, 1, 1]])
    b = csr_matrix([[0, 1, 0], [1, 1, 0]])

    # a = csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(2, 3))
    # b = csr_matrix([[0, 1, 0], [1, 1, 1]])

    kunn = KUNN()
    combined = union_csr_matrices(a, b)

    result = csr_matrix([[1, 1, 0], [1, 1, 1]])
    assert a.shape == b.shape == combined.shape
    numpy.testing.assert_almost_equal(combined.todense(), result.todense())


def test_predict_k_1():
    kunn = KUNN(Ku=1, Ki=1)

    training_matrix = csr_matrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

    # values = [1, 1, 1, 1, 1, 1, 1]
    # users = [0, 0, 1, 1, 2, 2, 2]
    # items = [1, 2, 0, 2, 0, 1, 2]
    # training_matrix = csr_matrix((values, (users, items)), shape=(5, 3))

    kunn.fit(training_matrix)

    pred_matrix = csr_matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # values_pred = [1, 1, 1, 1]
    # users_pred = [3, 3, 4, 4]
    # items_pred = [0, 1, 1, 2]
    # pred_matrix = csr_matrix(
    #     (values_pred, (users_pred, items_pred)), shape=training_matrix.shape
    # )

    prediction = kunn.predict(pred_matrix)

    itemknn = kunn._fit_item_knn(training_matrix)
    userknn = kunn._fit_user_knn(pred_matrix)

    # Manual computation of the formulas in the paper
    # We'll compute similarity 3, 2
    u = 3
    i = 2

    ## USER SIMILARITY ##

    # use ItemKNN class to compute user neighbours
    # By fitting it on the transpose of the combination of the test and pred matrices
    # In this case training_matrix and pred matrix are fully disjunct
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
    c_v = training_matrix[v].nnz
    c_u = (training_matrix[u] + pred_matrix[u]).nnz

    # Second summation, over all items v and u have in common
    # user 3 and user 2 have 2 items in common (0 and 1)
    # u's interactions are not in the training_matrix,
    # so we need increase the count of occurences + 1
    c_j_0 = training_matrix[:, 0].nnz + 1
    c_j_1 = training_matrix[:, 1].nnz + 1

    # Compute user similarity.
    # 1/sqrt(cv) + sum(j in [0, 1] 1/sqrt(c(j)))
    s_u = (1 / c_v ** 0.5) * ((1 / c_j_0 ** 0.5) + (1 / c_j_1 ** 0.5))

    ## ITEM SIMILARITY ##

    # K = 1 -> argmax gives us the most similar item
    j = numpy.argmax(itemknn[i])
    assert j == 0

    # c(j) in paper, only need it for the one j
    # User u has interacted with item 0 -> R_u_j = 1
    c_j = training_matrix[:, j].nnz
    c_i = training_matrix[:, i].nnz

    # 2 users have seen both items 0 and 2 => user 1,2
    numpy.testing.assert_array_equal(
        training_matrix[:, j].multiply(
            training_matrix[:, i]).toarray().nonzero()[0],
        numpy.array([1, 2]),
    )

    # Compute history lengths of the two users
    c_v_0 = training_matrix[1].nnz
    c_v_2 = training_matrix[2].nnz

    # Compute item similarity
    # 1/sqrt(c_j) * sum(v in [1,2]) 1/sqrt(c(v))
    s_i = (1 / c_j ** 0.5) * ((1 / c_v_0 ** 0.5) + (1 / c_v_2) ** 0.5)

    ## FINAL SCORE ##
    # (s_u + s_i) / (sqrt(c(u)*c(i)))
    s_u_i = (s_u + s_i) / ((c_u * c_i) ** 0.5)

    numpy.testing.assert_almost_equal(prediction[3, 2], s_u_i)


def test_predict_k_2():
    kunn = KUNN(Ku=2, Ki=2)

    training_matrix = csr_matrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

    # values = [1, 1, 1, 1, 1, 1, 1]
    # users = [0, 0, 1, 1, 2, 2, 2]
    # items = [1, 2, 0, 2, 0, 1, 2]
    # training_matrix = csr_matrix((values, (users, items)), shape=(5, 3))

    kunn.fit(training_matrix)

    pred_matrix = csr_matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # values_pred = [1, 1, 1, 1]
    # users_pred = [3, 3, 4, 4]
    # items_pred = [0, 1, 1, 2]
    # pred_matrix = csr_matrix(
    #     (values_pred, (users_pred, items_pred)), shape=training_matrix.shape
    # )

    prediction = kunn.predict(pred_matrix)

    # Fit the KNN models
    itemknn = kunn._fit_item_knn(training_matrix)
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
    one_over_sqrt_v = {v: 1 / training_matrix[v].nnz ** 0.5 for v in V}
    # Compute 1/sqrt(c(j)) for each j
    one_over_sqrt_j = {j: 1 / training_matrix[j].nnz ** 0.5 for j in J}

    ## USER SIMILARITY ##
    # Iteratively compute the user sim
    score = 0
    for v in V:  # 1st sum : v in KNN(u)
        if training_matrix[v, i] != 0:  # R_v_i in the first sum.
            # Compute transitive part
            trans_sum = 0
            # For the history of user u we can look in the combined matrices
            for j in (pred_matrix[u] + training_matrix[u]).nonzero()[
                1
            ]:  # Second sum, with R_u_i = 1

                if training_matrix[v, j] != 0:  # R_v_j = 1 clause in sum

                    c_j = training_matrix[:, j].nnz

                    # "HACK" To count the interaction of user u in the pred matrix
                    if training_matrix[u, j] == 0:
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
            for v in training_matrix[:, j].nonzero()[
                0
            ]:  # Second sum, with R_v_j clause
                if training_matrix[v, i] != 0:  # R_v_i clause in second sum
                    c_v = training_matrix[v, :].nnz
                    trans_sum += 1 / c_v ** 0.5

            score += trans_sum * one_over_sqrt_j[j]

    item_sim = score

    ## FINAL SCORE ##
    final_score = (user_sim + item_sim) / (
        (pred_matrix[u] + training_matrix[u]).nnz * training_matrix[:, i].nnz
    ) ** 0.5

    numpy.testing.assert_almost_equal(prediction[u, i], final_score)
