import pytest
import numpy as np

from recpack.metrics import (
    NDCGK,
    DCGK,
    RecallK,
    RRK,
    CoverageK,
    IPSHitRateK,
    PrecisionK,
    IntraListDiversityK,
    SNIPSHitRateK
)

from recpack.metrics.base import ListwiseMetricK, ElementwiseMetricK, GlobalMetricK, MetricTopK, FittedMetric


@pytest.mark.parametrize("metric_cls", [DCGK, RecallK, PrecisionK, IPSHitRateK, SNIPSHitRateK])
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


def test_eliminate_zeros(X_true, X_pred):
    recall = RecallK(2)

    X_true_aft, X_pred_aft = recall.eliminate_empty_users(X_true, X_pred)

    assert X_true.shape[1] == X_true_aft.shape[1]
    assert 2 == X_true_aft.shape[0]
