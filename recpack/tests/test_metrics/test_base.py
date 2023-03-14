# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
from recpack.metrics import (
    CoverageK,
    DCGK,
    NDCGK,
    HitK,
    PercentileRanking,
    PrecisionK,
    RecallK,
    CalibratedRecallK,
    ReciprocalRankK,
)
from recpack.metrics.base import ListwiseMetricK, GlobalMetricK, ElementwiseMetricK
from recpack.util import get_top_K_ranks

K = 2


@pytest.mark.parametrize(
    "metric",
    [
        CoverageK(K=K),
        DCGK(K=K),
        NDCGK(K=K),
        HitK(K=K),
        PercentileRanking(),
        PrecisionK(K=K),
        RecallK(K=K),
        CalibratedRecallK(K=K),
        ReciprocalRankK(K=K),
    ],
)
def test_results_shapes(metric, X_pred, X_true):
    metric.calculate(X_true, X_pred)

    if isinstance(metric, GlobalMetricK):
        # 1 value
        assert metric.results.shape == (1, 1)

    elif isinstance(metric, ListwiseMetricK):
        # 1 row per user (user, score)
        assert metric.results.shape == (len(set(X_true.nonzero()[0])), 2)

    elif isinstance(metric, ElementwiseMetricK):
        # 1 row per top K entry in the recos (user, item, score)
        assert metric.results.shape == (get_top_K_ranks(X_pred, K).nnz, 3)
