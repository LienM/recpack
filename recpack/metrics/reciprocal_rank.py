import logging

from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_inverse_nonzero


logger = logging.getLogger("recpack")


class RRK(ListwiseMetricK):
    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        # resolve top K items per user
        y_pred_top_K = self.get_top_K_ranks(y_pred)

        # compute hits
        hits = y_pred_top_K.multiply(y_true)
        # Invert hit ranks
        inverse_ranks = sparse_inverse_nonzero(hits)
        # per user compute the max inverted rank of a hit
        self.scores_ = inverse_ranks.max(axis=1)
