# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.metrics.base import FittedMetric, ElementwiseMetricK
from scipy.sparse import csr_matrix
import numpy as np


def compute_hits(y_true, y_pred):
    # Compute hits matrix:
    hits = y_true.multiply(y_pred)
    # Binarise the hits matrix
    hits[hits > 0] = 1

    return hits


class IPSMetric(FittedMetric):

    """IPS metrics are a class of metrics,
    where the probability of a user interacting with an item
    (propensity) is taken into account.

    Each score is weighted by the inverse propensity
    of the user interacting with the item.

    Before using an IPSMetric, it should be fitted to the data.

    :param ip_cap: Maximum value of an inverse propensity.
    Used to avoid excessively large weights for items that are rarely interacted with.
    :type ip_cap: int
    """

    def __init__(self):
        super().__init__()
        self.item_prob_ = None
        self.ip_cap = 10000

    def fit(self, X: csr_matrix):
        """Fit the propensities for the X dataset.

        We make the strong assumption that each user
        has the same probability to interact with an item.

        .. math::

            p(i|u) = p(i) = \\frac{|\\{u| u\\in U, X_{u,i} > 0\\}|} {|X|}

        Inverse propensity higher than the ``ip_cap``, are set to ``ip_cap``,
        to avoid that items that are never interacted with dominate the metric.

        :param X: The interactions to base the propensity computation on.
                    Suggested to use the labels you are trying to predict as value,
                    since that is the target.
        :type X: scipy.sparse.csr_matrix
        """
        # Compute vector with propensities
        self.item_prob_ = X.sum(axis=0) / X.sum()
        self.inverse_propensities = 1 / self.item_prob_
        self.inverse_propensities[self.inverse_propensities == np.inf] = 0

        self.inverse_propensities[self.inverse_propensities > self.ip_cap] = self.ip_cap


class IPSHitRateK(ElementwiseMetricK, IPSMetric):
    """Computes a weighted hits metric, with hits weighted by the user, item propensity.

    Each hit is weighted with the item's inverse propensity for the user.

    Higher values are better, they indicate the algorithm is able to
    recommend more long-tail items for the user.

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        assert self.item_prob_ is not None

        hits = compute_hits(y_true, y_pred_top_K)

        self.scores_ = hits.multiply(self.inverse_propensities)
