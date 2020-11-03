from recpack.metrics.base import MetricTopK
from scipy.sparse import csr_matrix


class PercentileRanking(MetricTopK):
    """
    This is a global metric where the rating r_ui of y_true and the rank/position of this item
    in the ordered prediction list (for user u in y_pred) is taken into account.
    Lower values of this percentile-rank are more advisable, because it will indicate that the highly rated items
    are closer to the top of the recommendation list.

    It will be calculated according the following forumula:
        perc-rank = \sum_{u,i}{r_{ui} * rank_{ui}} / \sum_{u,i}{r_{ui}}
    """

    # TODO: It is actually a global metric without calculate parameters
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
        ranking = self.get_top_K_ranks(csr_matrix(y_pred))

        # Ranking starts at 0 for this metric
        ranking.data -= 1
        rank_values = ranking / self.num_items_  # to get a percentile ranking
        numerator = y_true.multiply(rank_values).sum()

        self.value_ = numerator / denominator
        return
