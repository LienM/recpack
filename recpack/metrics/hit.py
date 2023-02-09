# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero

logger = logging.getLogger("recpack")


class HitK(ElementwiseMetricK):
    """Computes the number of hits in a list of Top-K recommendations.

    A hit is counted when a recommended item in the top K for this user was interacted with.

    Detailed :attr:`results` show which of the items in the list of Top-K recommended items
    were hits and which were not.

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1

        scores = scores.tocsr()

        self.scores_ = scores


class DiscountedGainK(ElementwiseMetricK):
    """Computes the discounted gain of every item in the Top-K recommendations of a user.

    Relevant items that are ranked higher in the Top-K recommendations have a higher gain.

    Detailed :attr:`results` show the gain of each item in the
    list of Top-K recommended items for every user.

    For each item :math:`i \\in \\text{TopK}(u)` the discounted gain is computed as

    .. math::

        \\text{DiscountedGain(u,i)} = \\frac{y^{true}_{u,i}}{\\log_2(\\text{rank}(u,i) + 1)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
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
