# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
import itertools

from scipy.sparse import csr_matrix
import numpy as np

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero


logger = logging.getLogger("recpack")


class DCGK(ListwiseMetricK):
    """Computes the sum of gains of all items in a recommendation list.

    Relevant items that are ranked higher in the Top-K recommendations have a higher gain.

    The Discounted Cumulative Gain (DCG) is computed for every user as

    .. math::

        \\text{DiscountedCumulativeGain}(u) = \\sum\\limits_{i \\in Top-K(u)} \\frac{y^{true}_{u,i}}{\\log_2 (\\text{rank}(u,i) + 1)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self.scores_ = csr_matrix(dcg.sum(axis=1))

        return


def dcg_k(y_true, y_pred, k=50):
    """Wrapper function around DiscountedCumulativeGain class.

    :param y_true: True labels
    :type y_true: csr_matrix
    :param y_pred: Predicted scores
    :type y_pred: csr_matrix
    :param k: Size of the recommendation list consisting of the Top-K item predictions.
    :type k: int, optional
    :return: global dcg value
    :rtype: float
    """
    r = DCGK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class NDCGK(ListwiseMetricK):

    """Computes the normalized sum of gains of all items in a recommendation list.

    The normalized Discounted Cumulative Gain (nDCG) is similar to DCG,
    but normalizes by dividing the resulting sum of cumulative gains
    by the best possible discounted cumulative gain for a list of recommendations
    of length K for a user with history length N.

    Scores are always in the interval [0, 1]

    .. math::

        \\text{NormalizedDiscountedCumulativeGain}(u) = \\frac{\\text{DCG}(u)}{\\text{IDCG}(u)}

    where IDCG stands for Ideal Discounted Cumulative Gain, computed as:

    .. math::

        \\text{IDCG}(u) = \\sum\\limits_{j=1}^{\\text{min}(K, |y^{true}_u|)} \\frac{1}{\\log_2 (j + 1)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))
        # Calculate IDCG values by creating a list of partial sums (the
        # functional way)
        self.IDCG_cache = np.array([1] + list(itertools.accumulate(self.discount_template, lambda x, y: x + y)))

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        # Correct predictions only
        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        per_user_dcg = dcg.sum(axis=1)

        hist_len = y_true.sum(axis=1).astype(np.int32)
        hist_len[hist_len > self.K] = self.K

        self.scores_ = sparse_divide_nonzero(
            csr_matrix(per_user_dcg),
            csr_matrix(self.IDCG_cache[hist_len]),
        )

        return


def ndcg_k(y_true, y_pred, k=50):
    """Wrapper function around NormalizedDiscountedCumulativeGain class.

    :param y_true: True labels
    :type y_true: csr_matrix
    :param y_pred: Predicted scores
    :type y_pred: csr_matrix
    :param k: Size of the recommendation list consisting of the Top-K item predictions.
    :type k: int, optional
    :return: ndcg value
    :rtype: float
    """
    r = NDCGK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
