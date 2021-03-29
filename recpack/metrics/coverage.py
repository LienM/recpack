from scipy.sparse import csr_matrix

from recpack.metrics.base import GlobalMetricK
from recpack.util import get_top_K_ranks


class CoverageK(GlobalMetricK):
    """Fraction of all items in ``y_true``and ``y_pred``
       that are ranked among the top-k predictions for any user.

    Computed as

    .. math::

        \\frac{|\\{i | \\exists u \\in U, i \\in \\text{TopK}(u) \\}|}{|I|}

    :param K: How many recommendations to consider for the metric.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        self.covered_items_ = set(y_pred_top_K.nonzero()[1])

        self.value_ = len(self.covered_items_) / self.num_items

        return
