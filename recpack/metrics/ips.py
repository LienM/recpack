from recpack.metrics.base import FittedMetric, ElementwiseMetricK
from scipy.sparse import csr_matrix
import numpy as np

from recpack.util import get_top_K_ranks


def compute_hits(y_true, y_pred):
    # Compute hits matrix:
    hits = y_true.multiply(y_pred)
    # Binarise the hits matrix
    hits[hits > 0] = 1

    return hits


class IPSMetric(FittedMetric):

    """IPS metrics are a class of metrics,
    where the user interaction probability is taken into account.

    Each score per item is weighted by the inverse propensity
    of the user interacting with the item.

    Before using an IPSMetric, it should be fitted to the data.

    :param ip_cap: used to avoid having an item that
        creates an incredible weight, any IP > ip_cap = ip_cap
    :type ip_cap: int
    """

    def __init__(self):
        super().__init__()
        self.item_prob_ = None
        self.ip_cap = 10000

    def fit(self, X: csr_matrix):
        """Fit the propensities for the X dataset

        We make the strong assumption that each user
        has the same probability to interact with an item.

        .. math::

            p(i|u) = p(i) = \\frac{|\\{u| u\\in U, X_{ui} > 0\\}|} {|X|}

        Inverse propensity higher than the ``ip_cap``, are set to ``ip_cap``,
        to avoid items seen almost never dominating the measure.

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
    """Computes a weighted hits per user metric.

    Each hit is weighted with the item's inverse propensity.
    Higher values are better, they indicate the algorithm is able to
    recommend more long tail items for the user.

    The value is aggregated by computing the average sum of IPS weighted hits per user.
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        assert self.item_prob_ is not None

        hits = compute_hits(y_true, y_pred_top_K)

        self.scores_ = hits.multiply(self.inverse_propensities)
