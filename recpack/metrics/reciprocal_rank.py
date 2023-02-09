# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging

from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_inverse_nonzero


logger = logging.getLogger("recpack")


class ReciprocalRankK(ListwiseMetricK):
    """Computes the inverse of the rank of the first hit
    in the recommendation list.

    The reciprocal rank for a user is calculated as

    .. math::

       \\text{ReciprocalRank}(u) = \\frac{1}{\\min\\limits_{i \\in \\text{Top-K}(u), \\\\ i \\in y^{true}_u} rank(u,i)}

    when there is at least one matching item between recommendations in :math:`\\text{Top-K}(u)` and targets in :math:`y^{true}_u`, 
    otherwise it is 0.

    :param K: Amount of top recommendations to consider in the metric calculation.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        # compute hits
        hits = y_pred_top_K.multiply(y_true)
        # Invert hit ranks
        inverse_ranks = sparse_inverse_nonzero(hits)
        # per user compute the max inverted rank of a hit
        self.scores_ = inverse_ranks.max(axis=1)
