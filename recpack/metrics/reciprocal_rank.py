import logging

from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_inverse_nonzero
from recpack.util import get_top_K_ranks

logger = logging.getLogger("recpack")


class ReciprocalRankK(ListwiseMetricK):
    """Reciprocal Rank or the inverse of the lowest rank of a hit.

    Per user with a hit the reciprocal rank is computed as

    .. math::

        \\text{RR}(u) = \\frac{1}{\\min\\limits_{i \\in KNN(u)} \\text{rank}(u,i) * y^{True}_{u,i}}
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
