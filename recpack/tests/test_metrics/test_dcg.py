# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np

from recpack.metrics.dcg import (
    DCGK,
    NDCGK,
    dcg_k,
    ndcg_k,
)


def test_dcgk_simple(X_pred, X_true_simplified):
    K = 2
    metric = DCGK(K)

    metric.calculate(X_true_simplified, X_pred)

    # rank 1  # rank 0  # 2 users
    expected_value = ((1 / np.log2(2 + 1)) + 1) / 2

    np.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcgk_simple(X_pred, X_true_simplified):
    K = 2
    metric = NDCGK(K)

    metric.calculate(X_true_simplified, X_pred)

    #  Number of true items for each user is one.
    IDCG = sum(1 / np.log2((i + 1) + 1) for i in range(0, 1))

    expected_value = ((1 / np.log2(2 + 1)) / IDCG + 1) / 2  # rank 1  # rank 0  # 2 users

    assert metric.num_users == 2
    np.testing.assert_almost_equal(metric.value, expected_value)


def test_dcgk(X_pred, X_true):
    K = 2
    metric = DCGK(K)

    metric.calculate(X_true, X_pred)

    # user 0 has 2 correct items, user 2 has two correct items, user 3 has no
    # predictions

    expected_value = (
        (1 / np.log2(2 + 1))
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1)))
    ) / 2  # 2 users

    np.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcg(X_pred, X_true):
    K = 2
    metric = NDCGK(K)

    metric.calculate(X_true, X_pred)

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / np.log2(3)

    expected_value = (
        (1 / np.log2(2 + 1)) / IDCG2
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1))) / IDCG2
    ) / 2  # 2 users

    assert metric.num_users == 2
    np.testing.assert_almost_equal(metric.value, expected_value)


def test_dcgk_3(X_pred, X_true):
    K = 3
    metric = DCGK(K)

    metric.calculate(X_true, X_pred)

    # user 0 has 2 correct items, user 2 has 2 correct items
    expected_value = (
        # user 2 rank 1  # user 2 rank 2
        (1 / np.log2(2 + 1) + 1 / np.log2(3 + 1))
        + (1 + (1 / np.log2(2 + 1)))  # user 0 rank 0  # user 0 rank 1
    ) / 2  # 2 users

    np.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcg_k3(X_pred, X_true):
    K = 3
    metric = NDCGK(K)

    metric.calculate(X_true, X_pred)

    # user 0 has 2 correct items, user 2 has three correct items
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / np.log2(3)
    IDCG3 = IDCG2 + 1 / np.log2(4)

    expected_value = (
        (1 + (1 / np.log2(3))) / IDCG2  # user 0 rank 0  # user 0 rank 1
        + (1 / np.log2(3) + 1 / np.log2(4)) / IDCG3  # user 2 rank 1  # user 2 rank 2
    ) / 2  # 2 users

    assert metric.num_users == 2
    np.testing.assert_almost_equal(metric.value, expected_value)


def test_dcgk_empty_reco(X_pred, X_true_unrecommended_user):
    K = 2
    metric = DCGK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    # user 0 has 2 correct items, user 2 has two correct items, user 3 has no
    # predictions

    expected_value = (
        (1 / np.log2(2 + 1))
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1)))
    ) / 3  # 3 users

    np.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcg_empty_reco(X_pred, X_true_unrecommended_user):
    K = 2
    metric = NDCGK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / np.log2(3)

    expected_value = (
        (1 / np.log2(2 + 1)) / IDCG2
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1))) / IDCG2
    ) / 3  # 3 users

    assert metric.num_users == 3
    np.testing.assert_almost_equal(metric.value, expected_value)


def test_dcgk_simple_func(X_pred, X_true_simplified):
    K = 2
    value = dcg_k(X_true_simplified, X_pred, K)

    # rank 1  # rank 0  # 2 users
    expected_value = ((1 / np.log2(2 + 1)) + 1) / 2

    np.testing.assert_almost_equal(value, expected_value)


def test_ndcgk_simple_func(X_pred, X_true_simplified):
    K = 2
    value = ndcg_k(X_true_simplified, X_pred, K)

    #  Number of true items for each user is one.
    IDCG = sum(1 / np.log2((i + 1) + 1) for i in range(0, 1))

    expected_value = ((1 / np.log2(2 + 1)) / IDCG + 1) / 2  # rank 1  # rank 0  # 2 users

    np.testing.assert_almost_equal(value, expected_value)


def test_dcgk_func(X_pred, X_true):
    K = 2
    value = dcg_k(X_true, X_pred, K)

    # user 0 has 2 correct items, user 2 has two correct items, user 3 has no
    # predictions

    expected_value = (
        (1 / np.log2(2 + 1))
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1)))
    ) / 2  # 2 users

    np.testing.assert_almost_equal(value, expected_value)


def test_ndcg_func(X_pred, X_true):
    K = 2
    value = ndcg_k(X_true, X_pred, K)

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / np.log2(3)

    expected_value = (
        (1 / np.log2(2 + 1)) / IDCG2
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1))) / IDCG2
    ) / 2  # 2 users

    np.testing.assert_almost_equal(value, expected_value)


def test_dcgk_3_func(X_pred, X_true):
    K = 3
    value = dcg_k(X_true, X_pred, K)

    # user 0 has 2 correct items, user 2 has 2 correct items
    expected_value = (
        # user 2 rank 1  # user 2 rank 2
        (1 / np.log2(2 + 1) + 1 / np.log2(3 + 1))
        + (1 + (1 / np.log2(2 + 1)))  # user 0 rank 0  # user 0 rank 1
    ) / 2  # 2 users

    np.testing.assert_almost_equal(value, expected_value)


def test_ndcg_k3_func(X_pred, X_true):
    K = 3
    value = ndcg_k(X_true, X_pred, K)

    # user 0 has 2 correct items, user 2 has three correct items
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / np.log2(3)
    IDCG3 = IDCG2 + 1 / np.log2(4)

    expected_value = (
        (1 + (1 / np.log2(3))) / IDCG2  # user 0 rank 0  # user 0 rank 1
        + (1 / np.log2(3) + 1 / np.log2(4)) / IDCG3  # user 2 rank 1  # user 2 rank 2
    ) / 2  # 2 users

    np.testing.assert_almost_equal(value, expected_value)


def test_dcgk_empty_reco_func(X_pred, X_true_unrecommended_user):
    K = 2
    value = dcg_k(X_true_unrecommended_user, X_pred, K)

    # user 0 has 2 correct items, user 2 has two correct items, user 3 has no
    # predictions

    expected_value = (
        (1 / np.log2(2 + 1))
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1)))
    ) / 3  # 3 users

    np.testing.assert_almost_equal(value, expected_value)


def test_ndcg_empty_reco_func(X_pred, X_true_unrecommended_user):
    K = 2
    value = ndcg_k(X_true_unrecommended_user, X_pred, K)

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / np.log2(3)

    expected_value = (
        (1 / np.log2(2 + 1)) / IDCG2
        # user 0 rank 1  # user 2 rank 0  # user 2 rank 1
        + (1 + (1 / np.log2(2 + 1))) / IDCG2
    ) / 3  # 3 users

    np.testing.assert_almost_equal(value, expected_value)
