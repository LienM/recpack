import logging

from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_inverse_nonzero


logger = logging.getLogger("recpack")


class ReciprocalRankK(ListwiseMetricK):
    """Computes the inverse of the rank of the first hit
    in the recommendation list.

    The reciprocal rank is calculated as:

    .. math::

       \\text{ReciprocalRank}(u) = \\frac{1}{\\text{rank}_{u,i}}

    with

    .. math::

        \\text{rank}_{u,i} = \\min\\limits_{i \\in Top-K(u), \\\\ i \\in y^{true}_u} rank(u,i)

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
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
