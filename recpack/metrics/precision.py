import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK
from recpack.util import get_top_K_ranks

logger = logging.getLogger("recpack")


class PrecisionK(ListwiseMetricK):
    """Computes precision@K: number of correct predictions in the top K.

    Different than in the definition for some classification tasks,
    the recommender is expected to return K items,
    if it does not, the missing items are considered misses.

    Precision is computed per user, as

    .. math::

        \\text{precision}(u) = \\frac{\\sum\\limits_{i \\in \\text{topK}(u)} R_{u,i}}{K}

    To get the final result, the sum of average precision over all users is taken.

    23/3: Changed base class from ElementwiseMetricK to ListwiseMetricK.
    Precision is never considered per user item pair,
    but usually per list of recommendations.
    If you want to know which items were hits, we have the HitMetric available.
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1

        scores = scores.tocsr()

        self.scores_ = csr_matrix(scores.sum(axis=1)) / self.K

        return
