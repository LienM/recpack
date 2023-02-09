# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import numpy as np
import numpy.random
from scipy.sparse import csr_matrix

from recpack.metrics import (
    NDCGK,
    DCGK,
    RecallK,
    ReciprocalRankK,
    CoverageK,
    IPSHitRateK,
    PrecisionK,
    IntraListDiversityK,
    HitK,
    DiscountedGainK,
)

from recpack.metrics.base import (
    ListwiseMetricK,
    ElementwiseMetricK,
    GlobalMetricK,
    MetricTopK,
    FittedMetric,
)


@pytest.mark.parametrize("metric_cls", [HitK, IPSHitRateK, DiscountedGainK])
def test_results_elementwise_topK(metric_cls, X_true, X_pred):
    K = 2

    metric = metric_cls(K)

    if isinstance(metric, FittedMetric):
        metric.fit(X_pred)

    metric.calculate(X_true, X_pred)

    assert hasattr(metric, "scores_")

    results = metric.results

    # TODO verify something about the shape
    np.testing.assert_array_equal(results.columns, ["user_id", "item_id", "score"])

    assert 2 in results["user_id"].unique()
    assert 1 not in results["user_id"].unique()

    assert results.shape[0] == metric.num_users_ * K  # K interactions for each user
    assert metric.num_users_ == X_pred.sum(axis=1)[:, 0].nonzero()[0].shape[0]
    assert metric.num_items_ == X_pred.shape[1]


@pytest.mark.parametrize(
    "metric_cls",
    [
        DCGK,
        RecallK,
        PrecisionK,
        NDCGK,
        ReciprocalRankK,
        IntraListDiversityK,
    ],
)
def test_results_listwise_topK(metric_cls, X_true, X_pred):
    K = 2

    metric = metric_cls(K)

    if isinstance(metric, FittedMetric):
        metric.fit(X_pred)

    metric.calculate(X_true, X_pred)

    assert hasattr(metric, "scores_")

    results = metric.results
    np.testing.assert_array_equal(results.columns, ["user_id", "score"])

    assert 2 in results["user_id"].unique()
    assert 1 not in results["user_id"].unique()

    assert results.shape[0] == metric.num_users_  # One entry for each user
    assert metric.num_users_ == X_pred.sum(axis=1)[:, 0].nonzero()[0].shape[0]
    assert metric.num_items_ == X_pred.shape[1]


def test_eliminate_zeros(X_true, X_pred):
    recall = RecallK(2)

    X_true_aft, X_pred_aft = recall._eliminate_empty_users(X_true, X_pred)

    assert X_true.shape[1] == X_true_aft.shape[1]
    assert 2 == X_true_aft.shape[0]


@pytest.mark.parametrize("metric_cls", [HitK, IPSHitRateK, DiscountedGainK])
def test_results_elementwise_topK_no_reco(metric_cls, X_true_unrecommended_user, X_pred):
    K = 2

    metric = metric_cls(K)

    if isinstance(metric, FittedMetric):
        metric.fit(X_pred)

    metric.calculate(X_true_unrecommended_user, X_pred)

    assert hasattr(metric, "scores_")

    results = metric.results

    # TODO verify something about the shape
    np.testing.assert_array_equal(results.columns, ["user_id", "item_id", "score"])

    assert 2 in results["user_id"].unique()
    assert 1 not in results["user_id"].unique()

    assert results.shape[0] == metric.num_users_ * K  # K interactions for each user
    # There is a user without any predictions,
    # so the number of users is equal to
    # 1 + the number of users with predictions
    assert metric.num_users_ == X_pred.sum(axis=1)[:, 0].nonzero()[0].shape[0] + 1
    assert metric.num_items_ == X_pred.shape[1]


@pytest.mark.parametrize(
    "metric_cls",
    [
        DCGK,
        RecallK,
        PrecisionK,
        NDCGK,
        ReciprocalRankK,
        IntraListDiversityK,
    ],
)
def test_results_listwise_topK_no_reco(metric_cls, X_true_unrecommended_user, X_pred):
    K = 2

    metric = metric_cls(K)

    if isinstance(metric, FittedMetric):
        metric.fit(X_pred)

    metric.calculate(X_true_unrecommended_user, X_pred)

    assert hasattr(metric, "scores_")

    results = metric.results
    np.testing.assert_array_equal(results.columns, ["user_id", "score"])

    assert 2 in results["user_id"].unique()
    assert 1 not in results["user_id"].unique()

    assert results.shape[0] == metric.num_users_  # One entry for each user
    # There is a user without any prediction
    assert metric.num_users_ == X_pred.sum(axis=1)[:, 0].nonzero()[0].shape[0] + 1
    assert metric.num_items_ == X_pred.shape[1]


def test_eliminate_zeros_no_reco(X_true_unrecommended_user, X_pred):
    recall = RecallK(2)

    X_true_aft, X_pred_aft = recall._eliminate_empty_users(X_true_unrecommended_user, X_pred)

    assert X_true_unrecommended_user.shape[1] == X_true_aft.shape[1]
    assert 3 == X_true_aft.shape[0]


def test_verify_shapes():
    m = DCGK(3)

    y_pred = csr_matrix(np.ones((3, 2)))
    y_true = csr_matrix(np.ones((2, 3)))

    with pytest.raises(AssertionError):
        m.calculate(y_true, y_pred)
