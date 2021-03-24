import logging

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero
from recpack.util import get_top_K_ranks

logger = logging.getLogger("recpack")


class HitK(ElementwiseMetricK):
    """Metric computing the hits in a prediction list.

    Each user, item pair has score 0 or 1,
    1 if it is both in the topK of recommended scores and the true labels matrix.

    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1

        scores = scores.tocsr()

        self.scores_ = scores


# TODO: Tests
class WeightedHitK(ElementwiseMetricK):
    """Hit metric, with each hit weighted by the number of interactions of that user.

    For users with more items it is "easier" to predict an item correctly,
    so in detailed analysis it is interesting to consider the weighted result.

    For each item :math:`i \\in TopK(u)` the discounted gain is computed as.

    .. math::

        \\frac{y^{true}_{u,i}}{\\sum_{j \\in I} y^{true}_{u,j}}

    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1

        scores = scores.tocsr()

        self.scores_ = sparse_divide_nonzero(scores, csr_matrix(y_true.sum(axis=1)))


class DiscountedGainK(ElementwiseMetricK):
    """Discounted gain, hits weighted by the inverse of their rank.

    Hits at lower positions have a higher chance of getting seen by users,
    and as such are more important.

    For each item :math:`i \\in \\text{TopK}(u)` the discounted gain is computed as.

    .. math::

        \\frac{y^{true}_{u_i}}{\\log_2(\\text{rank}(u,i) + 1)}
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self.scores_ = dcg

        return
