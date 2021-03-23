import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero
from recpack.util import get_top_K_ranks

logger = logging.getLogger("recpack")


class RecallK(ListwiseMetricK):
    """Recall, as the fraction of relevant items retrieved in top K.

    Recall per user computed as

    .. math::

        \\text{Recall}(u) = \\frac{\\sum_{i \\in \\text{topK}(u)} R_{u,i} * P_{u,i}}{\\sum_{j \\in I} R_{u,j}}


    To get a single result, the mean of recall per user is computed.

    23/3: Changed from ElementwiseMetricK to ListwiseMetricK,
    recall is always discussed per user.
    Because it is still interesting to get a weighted hit value,
    with the number of items seen by the user,
    added a new metric WeightedHitMetric.
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1

        scores = scores.tocsr()

        self.scores_ = csr_matrix(
            sparse_divide_nonzero(scores, csr_matrix(y_true.sum(axis=1))).sum(axis=1)
        )

        return


def recall_k(y_true, y_pred, k=50):
    r = RecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class CalibratedRecallK(ListwiseMetricK):
    """Recall as the number of retrieved positives divided
    by the minimum of K and number of relevant items for the user.

    This differs from normal recall in that it accounts for when K < #relevant,
    because we can't expect a list of K items to cover more than K items.
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        scores = scores.tocsr()

        optimal = csr_matrix(np.minimum(y_true.sum(axis=1), self.K))

        self.scores_ = sparse_divide_nonzero(scores, optimal).sum(axis=1)


def calibrated_recall_k(y_true, y_pred, k):
    r = CalibratedRecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
