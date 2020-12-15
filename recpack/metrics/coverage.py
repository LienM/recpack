from scipy.sparse import csr_matrix

from recpack.metrics.base import GlobalMetricK
from recpack.util import get_top_K_ranks


class CoverageK(GlobalMetricK):
    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        top_k_pred = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = top_k_pred

        self.covered_items_ = set(top_k_pred.nonzero()[1])

        self.value_ = len(self.covered_items_) / self.num_items

        return
