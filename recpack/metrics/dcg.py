import logging
import itertools

from scipy.sparse import csr_matrix
import numpy as np

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero
from recpack.util import get_top_K_ranks

logger = logging.getLogger("recpack")


class DCGK(ListwiseMetricK):
    """Discounted Cumulative Gain metric. Sum of cumulative gains.

    Discounted Cumulative Gain is computed as follows,

    .. math::

        DCG(u) = \\sum_{i \\in TopK(u)} \\frac{y^{true}_{u,i}}{\\log_2 (rank(u,i) + 1)}

    NDCG is then the sum of DG for the whole list.

    :param ElementwiseMetricK: [description]
    :type ElementwiseMetricK: [type]
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self.scores_ = csr_matrix(dcg.sum(axis=1))

        return


def dcg_k(y_true, y_pred, k=50):
    r = DCGK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class NDCGK(ListwiseMetricK):
    """Normalized Discounted Cumulative Gain metric.

    NDCG is similar to DCG, but normalises by dividing with the optimal,
    possible DCG for the recommendation.
    Thus accounting for users where less than K items are available,
    and so the max score is lower than for other users.

    Scores are always in the interval [0, 1]

    :param K: How many of the top recommendations to consider.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))
        # Calculate IDCG values by creating a list of partial sums (the
        # functional way)
        self.IDCG_cache = np.array(
            [1] + list(itertools.accumulate(self.discount_template, lambda x, y: x + y))
        )

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        # Correct predictions only
        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        per_user_dcg = dcg.sum(axis=1)

        hist_len = y_true.sum(axis=1).astype(np.int32)
        hist_len[hist_len > self.K] = self.K

        self.scores_ = sparse_divide_nonzero(
            csr_matrix(per_user_dcg), csr_matrix(self.IDCG_cache[hist_len])
        )

        return


def ndcg_k(y_true, y_pred, k=50):
    """Wrapper function around ndcg class.

    :param y_true: True labels
    :type y_true: csr_matrix
    :param y_pred: Predicted scores
    :type y_pred: csr_matrix
    :param k: top k to use for prediction, defaults to 50.
    :type k: int, optional
    :return: ndcg value
    :rtype: float
    """
    r = NDCGK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
