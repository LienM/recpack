import logging

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero


logger = logging.getLogger("recpack")


class RecallK(ElementwiseMetricK):
    """
    Calculates the recall as follows:
    Recall@K = #relevant in top K / #relevant
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1

        scores = scores.tocsr()

        self.scores_ = sparse_divide_nonzero(scores, csr_matrix(y_true.sum(axis=1)))

        return


def recall_k(y_true, y_pred, k=50):
    r = RecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class CalibratedRecallK(ElementwiseMetricK):
    """
    Calculates the recall as follows:
    Recall@K = #relevant in top K / min(K, #relevant)

    This differs from normal recall in that it accounts for when K < #relevant,
    resulting in the value being normalized to the range [0, 1].
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        scores = scores.tocsr()

        optimal = csr_matrix(np.minimum(y_true.sum(axis=1), self.K))

        self.scores_ = sparse_divide_nonzero(scores, optimal)

        return


def calibrated_recall_k(y_true, y_pred, k):
    r = CalibratedRecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
