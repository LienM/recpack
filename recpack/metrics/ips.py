from recpack.metrics.base import FittedMetric, ElementwiseMetricK
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
import numpy as np


def compute_hits(y_true, y_pred):
    # Compute hits matrix:
    hits = y_true.multiply(y_pred)
    # Binarise the hits matrix
    hits[hits > 0] = 1

    return hits


class IPSMetric(FittedMetric):

    """IPS metrics are a class of metrics, where the user interaction probability is taken into account.

    Each score per item is weighted by the inverse propensity of the user interacting with the item.
    :param ip_cap: used to avoid having an item that creates an incredible weight, any IP > ip_cap = ip_cap
    :type ip_cap: int
    """

    def __init__(self, ip_cap=10000):
        super().__init__()
        self.item_prob_ = None
        self.ip_cap = ip_cap

    def fit(self, X: csr_matrix):
        """Fit the propensities for the X dataset

        We make the very strong assumption that each user has the same probability to rate an item.
        If we have more data about users, we could use classification to improve the probability.

        For now
        pi(i) = # interactions with i / # interactions

        :param X: The interactions to base the propensity computation on.
                    Suggested to use X_true as value for this, since that is the target.
        :type X: scipy.sparse.csr_matrix
        """
        # Compute vector with propensities
        self.item_prob_ = X.sum(axis=0) / X.sum()
        self.inverse_propensities = 1 / self.item_prob_
        self.inverse_propensities[self.inverse_propensities == np.inf] = 0

        self.inverse_propensities[self.inverse_propensities > self.ip_cap] = self.ip_cap


class IPSHitRateK(ElementwiseMetricK, IPSMetric):
    """
    Computes a weighted hits per user metric.

    Each hit is weighted with the item's inverse propensity.
    The value is aggregated by summing per user, and taking the average
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        assert self.item_prob_ is not None
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        # Per user get a set of the topK predicted items
        y_pred_top_K = self.get_top_K_ranks(y_pred)

        hits = compute_hits(y_true, y_pred_top_K)

        self.scores_ = hits.multiply(self.inverse_propensities)

        self.value_ = self.scores_.sum() / (self.num_users)


class SNIPSHitRateK(IPSHitRateK):
    """
    Self Normalizing IPS hit rate.

    each weight per item is normalised by the maximal possible score for the user.

    The reason this is done,
    is that the IPS metric can create a bias towards infrequently shown items,
    by normalizing this bias can be removed.
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        assert self.item_prob_ is not None
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        # Per user get a set of the topK predicted items
        y_pred_top_K = self.get_top_K_ranks(y_pred)

        hits = compute_hits(y_true, y_pred_top_K)

        # Compute max possible score per user, to normalise
        propensities_per_hit = y_true.multiply(self.inverse_propensities).tocsr()

        top_k_mask = self.get_top_K_ranks(propensities_per_hit)
        top_k_mask[top_k_mask > 0] = 1

        max_prop_possible = propensities_per_hit.multiply(top_k_mask)

        m = max_prop_possible.sum(axis=1)

        # We can safely take the inverse,
        # because none of the hits should ever have propensity 0 if properly computed
        self.scores_ = hits.multiply(self.inverse_propensities).multiply(1 / m)

        self.value_ = self.scores_.sum() / (self.num_users)
