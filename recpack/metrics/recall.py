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

        \\text{Recall}(u) = \\frac{\\sum\\limits_{i \\in \\text{topK}(u)} y^{true}_{u,i} }{\\sum\\limits_{j \\in I} y^{true}_{u,j}}

    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

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

    .. math::

        \\text{Recall}(u) = \\frac{\\sum\\limits_{i \\in \\text{topK}(u)} y^{true}_{u,i} }{\\text{min}(\\sum\\limits_{j \\in I} y^{true}_{u,j}, K)}


    This differs from normal recall in that it accounts for when K < #relevant,
    because we can't expect a list of K items to cover more than K items.
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        scores = scores.tocsr()

        optimal = csr_matrix(np.minimum(y_true.sum(axis=1), self.K))

        self.scores_ = sparse_divide_nonzero(scores, optimal).sum(axis=1)


def calibrated_recall_k(y_true, y_pred, k):
    r = CalibratedRecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
