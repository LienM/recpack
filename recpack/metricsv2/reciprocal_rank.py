import scipy.sparse
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from recpack.metricsv2.base import ListwiseMetric, MetricTopK
from recpack.metricsv2.util import sparse_inverse_nonzero
class RRK(ListwiseMetric, MetricTopK):
    def __init__(self, K):
        MetricTopK.__init__(self, K)

        self.col_names = ["user", "score"]
        self.rr = None

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
        
        # MRR
        self._value = self.scores_.mean()

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame(dict(zip(self.col_names, scipy.sparse.find(self.scores_))))
