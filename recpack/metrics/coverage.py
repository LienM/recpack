# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from scipy.sparse import csr_matrix

from recpack.metrics.base import GlobalMetricK


class CoverageK(GlobalMetricK):
    """Fraction of all items that are ranked among the
    Top-K recommendations for any user.

    Computed as

    .. math::

        \\frac{|\\{i \\in I | (\\exists u \\in U) [i \\in \\text{TopK}(u)] \\}|}{|I|}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        self.covered_items_ = set(y_pred_top_K.nonzero()[1])

        self.value_ = len(self.covered_items_) / self.num_items

        return
