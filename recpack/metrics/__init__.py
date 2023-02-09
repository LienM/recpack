# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""

The metrics module in recpack contains a large amount of metrics
commonly used to evaluate recommendation algorithms.

All metrics assume that we have access to a set ``y_true`` of true
user interactions that we are trying to predict and a set of
recommendation scores ``y_pred``.
We can then evaluate how well our algorithm was able to predict
these interactions in ``y_true``.

Most metrics are "Top-K Metrics": they consider
only the Top-K best scoring item predictions, as the number of
recommendations that can be shown in a realistic setting
is limited.

.. currentmodule:: recpack.metrics

.. contents:: Table of Contents
    :depth: 2

Global Metrics
---------------

A global metric reports only a single, global metric value.


.. autosummary::
    :toctree: generated/

    CoverageK
    PercentileRanking

Listwise Metrics
----------------

A listwise metric reports one metric value for every user.
To obtain a global metric value, these per-user scores are averaged.


.. autosummary::
    :toctree: generated/

    DCGK
    NDCGK
    RecallK
    CalibratedRecallK
    PrecisionK
    ReciprocalRankK

Elementwise Metric
------------------

An elementwise metric reports a score for every user-item pair in the Top-K.
To obtain a global metric value, these scores are summed per user, then averaged.

.. autosummary::
    :toctree: generated/

    HitK
    DiscountedGainK
"""


from recpack.metrics.coverage import CoverageK
from recpack.metrics.dcg import (
    DCGK,
    NDCGK,
)
from recpack.metrics.diversity import IntraListDiversityK
from recpack.metrics.hit import HitK, DiscountedGainK
from recpack.metrics.ips import IPSHitRateK
from recpack.metrics.precision import PrecisionK
from recpack.metrics.recall import RecallK, CalibratedRecallK
from recpack.metrics.reciprocal_rank import ReciprocalRankK
from recpack.metrics.percentile_ranking import PercentileRanking

METRICS = {
    "CoverageK": CoverageK,
    "NDCGK": NDCGK,
    "DCGK": DCGK,
    "IntraListDiversityK": IntraListDiversityK,
    "IPSHitRateK": IPSHitRateK,
    "HitK": HitK,
    "DiscountedGainK": DiscountedGainK,
    "PrecisionK": PrecisionK,
    "RecallK": RecallK,
    "CalibratedRecallK": CalibratedRecallK,
    "ReciprocalRankK": ReciprocalRankK,
    "PercentileRanking": PercentileRanking,
}
