import pytest
import numpy as np
import numpy.random
import scipy.sparse

from recpack.metrics import (
    NDCGK,
    DCGK,
    RecallK,
    RRK,
    CoverageK,
    IPSHitRateK,
    PrecisionK,
    IntraListDiversityK,
)

from recpack.metrics.base import (
    ListwiseMetricK,
    ElementwiseMetricK,
    GlobalMetricK,
    MetricTopK,
    FittedMetric,
)


@pytest.mark.parametrize(
    "metric_cls", [DCGK, RecallK, PrecisionK, IPSHitRateK]
)
def test_results_elementwise_topK(metric_cls, X_true, X_pred):
    K = 2

    metric = metric_cls(K)

    if isinstance(metric, FittedMetric):
        metric.fit(X_pred)

    metric.calculate(X_true, X_pred)

    assert hasattr(metric, "scores_")

    results = metric.results

    # TODO verify something about the shape
    np.testing.assert_array_equal(
        results.columns, [
            "user_id", "item_id", "score"])

    assert 2 in results["user_id"].unique()
    assert 1 not in results["user_id"].unique()

    assert results.shape[0] == metric.num_users_ * \
        K  # K interactions for each user
    # There is a user without any predictions,
    # so the number of users is equal to
    # 1 + the number of users with predictions
    assert metric.num_users_ == X_pred.sum(
        axis=1)[:, 0].nonzero()[0].shape[0] + 1
    assert metric.num_items_ == X_pred.shape[1]


@pytest.mark.parametrize("metric_cls", [NDCGK, RRK, IntraListDiversityK])
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
    # There is a user without any prediction
    assert metric.num_users_ == X_pred.sum(
        axis=1)[:, 0].nonzero()[0].shape[0] + 1
    assert metric.num_items_ == X_pred.shape[1]


def test_get_topK_ranks():
    state = np.random.RandomState(13940)

    mat = scipy.sparse.random(2, 100, density=0.10, random_state=state).tocsr()

    metric = MetricTopK(20)

    top_k_ranks = metric.get_top_K_ranks(mat)

    max_ix_row_0 = np.argmax(mat[0, :])
    assert top_k_ranks[0, max_ix_row_0] == 1

    max_ix_row_1 = np.argmax(mat[1, :])
    assert top_k_ranks[1, max_ix_row_1] == 1


def test_eliminate_zeros(X_true, X_pred):
    recall = RecallK(2)

    X_true_aft, X_pred_aft = recall.eliminate_empty_users(X_true, X_pred)

    assert X_true.shape[1] == X_true_aft.shape[1]
    assert 3 == X_true_aft.shape[0]
