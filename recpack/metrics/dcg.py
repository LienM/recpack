import logging
import itertools

from scipy.sparse import csr_matrix
import numpy as np

from recpack.metrics.base import ElementwiseMetricK, ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero


logger = logging.getLogger("recpack")


class DCGK(ElementwiseMetricK):
    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self.scores_ = dcg

        return


def dcg_k(y_true, y_pred, k=50):
    r = DCGK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class NDCGK(ListwiseMetricK):
    def __init__(self, K):
        super().__init__(K)

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))
        # Calculate IDCG values by creating a list of partial sums (the
        # functional way)
        self.IDCG_cache = np.array(
            [1] + list(itertools.accumulate(self.discount_template, lambda x, y: x + y))
        )

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

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
            csr_matrix(per_user_dcg), csr_matrix(self.IDCG_cache[hist_len])
        )

        return


def ndcg_k(y_true, y_pred, k=50):
    r = NDCGK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
