from scipy.sparse import csr_matrix

from recpack.metrics.base import GlobalMetricK
from recpack.util import get_top_K_ranks


class PercentileRanking(GlobalMetricK):
    """Expected Percentile Ranking.

    Metric as described in Hu, Yifan, Yehuda Koren, and Chris Volinsky.
    "Collaborative filtering for implicit feedback datasets."
    2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.

    Percentile ranking is calculated according the following forumula:

    .. math::

        \\text{perc_rank} = \\frac{\\sum_{u \\in U,i \\in I}y^{true}_{ui} * \\overline{\\text{rank}}_{ui}}{\\sum_{u \\in U,i \\in I} y^{true}_{ui}}

    where

    .. math::

        \\overline{rank}_{ui} =
        \\begin{cases}
            \\frac{\\text{rank}_{ui} - 1}{|I|} & \\text{if $i$ $\\in$ TopK($u$)}\\\\
            1 & \\text{otherwise}
        \\end{cases}

    Lower values indicate
    in the ordered prediction list (for user u in y_pred) is taken into account.
    Lower values of this percentile-rank are more advisable,
    because it will indicate that the highly rated items
    are closer to the top of the recommendation list.
    """

    def __init__(self):
        super().__init__(None)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """
        Calculate the percentile ranking score for the particular y_true and y_pred matrices.
        :param y_true: User-item matrix with the actual true rating values.
        :param y_pred: User-item matrix with all prediction rating scores.
        :return: None: The result will be saved in self.value.
        """
        self.num_users_, self.num_items_ = y_true.shape
        self.K = self.num_items_
        self.verify_shape(y_true, y_pred)

        denominator = y_true.sum()
        ranking = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = ranking

        # Ranking starts at 0 for this metric
        ranking.data -= 1
        rank_values = ranking / self.num_items_  # to get a percentile ranking

        # Compute the percentile rankings of hits in the topK
        hit_mat = y_true.multiply(rank_values)

        # Add 1 for each y_true value not in the topK
        hit_mat.data = hit_mat.data - 1
        ranking_mat = y_true + hit_mat

        # Numerator is the sum of the ranks of y_true values.
        numerator = ranking_mat.sum()

        self.value_ = numerator / denominator
        return
