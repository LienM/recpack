import numpy
import pytest
import scipy.sparse

import recpack.metrics as metrics


def test_recall(X_pred, X_true):
    K = 2
    metric = metrics.RecallK(K)

    metric.update(X_pred, X_true)

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, 0.75)


def test_recall_2(X_pred, X_true):
    K = 2
    metric = metrics.RecallK(K)

    scores = metric.fit_transform(X_pred, X_true)
    value = scores.per_user_average()

    numpy.testing.assert_almost_equal(value, 0.75)


def test_mrr(X_pred, X_true):
    K = 2
    metric = metrics.MeanReciprocalRankK(K)

    metric.update(X_pred, X_true)

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, 0.75)


def test_mrr_2(X_pred, X_true):
    K = 2
    metric = metrics.MeanReciprocalRankK(K)

    scores = metric.fit_transform(X_pred, X_true)
    value = scores.per_user_average()

    numpy.testing.assert_almost_equal(value, 0.75)


def test_ndcg_simple(X_pred, X_true_simplified):
    K = 2
    metric = metrics.NDCGK(K)

    metric.update(X_pred, X_true_simplified)

    #  Number of true items for each user is one.
    IDCG = sum(1 / numpy.log2((i + 1) + 1) for i in range(0, 1))

    expected_value = (
        (1 / numpy.log2(2 + 1)) / IDCG +    # rank 1
        1                                   # rank 0
    ) / 2  # 2 users

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcg(X_pred, X_true):
    K = 2
    metric = metrics.NDCGK(K)

    metric.update(X_pred, X_true)

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / numpy.log2(3)

    expected_value = (
            (1 / numpy.log2(2 + 1)) / IDCG2 +       # user 0 rank 1
            (
                1 +                                 # user 2 rank 0
                (1 / numpy.log2(2 + 1))             # user 2 rank 1
            ) / IDCG2
     ) / 2  # 2 users

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcg_k3(X_pred, X_true):
    K = 3
    metric = metrics.NDCGK(K)

    metric.update(X_pred, X_true)

    # user 0 has 2 correct items, user 2 has three correct items
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / numpy.log2(3)
    IDCG3 = IDCG2 + 1 / numpy.log2(4)

    expected_value = (
        (
            1 +                                 # user 0 rank 0
            (1 / numpy.log2(3))                 # user 0 rank 1
        ) / IDCG2 +
        (
            1 / numpy.log2(3) +                 # user 2 rank 1
            1 / numpy.log2(4)                   # user 2 rank 2
        ) / IDCG3
     ) / 2  # 2 users

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_value)


def test_ndcg_simple_2(X_pred, X_true_simplified):
    K = 2
    metric = metrics.NDCGK(K)

    scores = metric.fit_transform(X_pred, X_true_simplified)
    value = scores.per_user_average()

    #  Number of true items for each user is one.
    IDCG = sum(1 / numpy.log2((i + 1) + 1) for i in range(0, 1))

    expected_value = (
        (1 / numpy.log2(2 + 1)) / IDCG +    # rank 1
        1                                   # rank 0
    ) / 2  # 2 users

    numpy.testing.assert_almost_equal(value, expected_value)


def test_ndcg_2(X_pred, X_true):
    K = 2
    metric = metrics.NDCGK(K)

    scores = metric.fit_transform(X_pred, X_true)
    value = scores.per_user_average()

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / numpy.log2(3)

    expected_value = (
        (
            1 +                                 # user 0 rank 0
            (1 / numpy.log2(2 + 1))             # user 0 rank 1
        ) / IDCG2 +
        (1 / numpy.log2(2 + 1)) / IDCG2        # user 2 rank 1
     ) / 2  # 2 users

    numpy.testing.assert_almost_equal(value, expected_value)


def test_ndcg_2_k3(X_pred, X_true):
    K = 3
    metric = metrics.NDCGK(K)

    scores = metric.fit_transform(X_pred, X_true)
    value = scores.per_user_average()

    # user 0 has 2 correct items, user 2 has three correct items
    IDCG1 = 1
    IDCG2 = IDCG1 + 1 / numpy.log2(3)
    IDCG3 = IDCG2 + 1 / numpy.log2(4)

    expected_value = (
        (
            1 +                                 # user 0 rank 0
            (1 / numpy.log2(3))                 # user 0 rank 1
        ) / IDCG2 +
        (
            1 / numpy.log2(3) +                 # user 2 rank 1
            1 / numpy.log2(4)                   # user 2 rank 2
        ) / IDCG3
     ) / 2  # 2 users

    numpy.testing.assert_almost_equal(value, expected_value)
