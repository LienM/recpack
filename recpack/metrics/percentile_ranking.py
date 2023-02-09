# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from scipy.sparse import csr_matrix

from recpack.metrics.base import Metric
from recpack.util import get_top_K_ranks
from recpack.matrix import to_binary


class PercentileRanking(Metric):
    """Expected Percentile Ranking.

    Metric as described in Hu, Yifan, Yehuda Koren, and Chris Volinsky.
    "Collaborative filtering for implicit feedback datasets."
    2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
    With a change to account for items that receive no recommendation score for a user.

    Percentile ranking is calculated according to the following formula:

    .. math::

        \\text{perc_rank} = \\frac{\\sum\\limits_{u \\in U,i \\in I} y^{true}_{u,i} * \\overline{\\text{rank}}_{u,i}}{\\sum\\limits_{u \\in U,i \\in I} y^{true}_{u,i}}

    where

    .. math::

        \\overline{rank}_{u,i} =
        \\begin{cases}
            \\frac{\\text{rank}_{u,i} - 1}{|I|} & \\text{if } i \\in y^{pred}(u) \\\\
            \\frac{\\max\\limits_{j} (\\text{rank}_{uj}) + |I|}{2|I|} & \\text{otherwise}
        \\end{cases}

    Non predicted items in the :math:`y^{true}` matrix,
    get the average rank from all remaining items per user.
    As if these remaining items would be ordered randomly.

    Lower values of this percentile-ranking are desirable,
    because that indicates relevant items are shown at higher positions.
    """

    def __init__(self):
        super().__init__()

    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Calculate the percentile ranking score for the particular
        ``y_true`` and ``y_pred`` matrices.

        Assumes a binary ``y_true`` matrix.

        :param y_true: User-item matrix with the actual true rating values.
        :param y_pred: User-item matrix with all prediction rating scores.
        :return: None: The result will be saved in self.value.
        """

        # Get ranks for all recommended items.
        # Items with 0 score, will get no rank.
        # This transformation can be quite expensive if the y_pred matrix is dense.
        K = self.num_items
        ranking = get_top_K_ranks(y_pred, K)

        # Ranking starts at 0 for this metric
        # ranking.data -= 1
        rank_values = ranking / self.num_items  # to get a percentile ranking
        rank_values.data = rank_values.data - (1 / self.num_items)  # Ranking starts at 0

        # Compute the percentile rankings of hits in the topK
        hit_mat = y_true.multiply(rank_values)

        # Account for items that were expected, but not recommended
        # These items will get the average between max recommended rank,
        # and max possible rank
        # This means they are randomly distributed in the group of all 0 scores,
        # but we use the expected value rather than giving each a random rank
        # to improve computation speed.
        max_rank_per_user = rank_values.max(axis=1)

        rank_for_misses_per_user = csr_matrix((max_rank_per_user.toarray() + 1) / 2)

        # Add the average rank for all non matches.
        pure_hit = y_true.multiply(y_pred)
        ranking_mat = (y_true - to_binary(pure_hit)).multiply(rank_for_misses_per_user) + hit_mat

        # Multiply with 100 to get percents io fractions
        ranking_mat = ranking_mat * 100

        # Numerator is the sum of the ranks of y_true values.
        numerator = ranking_mat.sum()
        denominator = y_true.sum()

        self.value_ = numerator / denominator
