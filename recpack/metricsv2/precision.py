import logging

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metricsv2.base import ElementwiseMetric, MetricTopK
from recpack.metricsv2.util import sparse_divide_nonzero


logger = logging.getLogger("recpack")


class PrecisionK(ElementwiseMetric, MetricTopK):
    def __init__(self, K):
        ElementwiseMetric.__init__(self)
        MetricTopK.__init__(self, K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        self.verify_shape(y_true, y_pred)

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1

        scores = scores.tocsr()

        self.scores_ = scores / self.K

        self._value = self.scores_.sum(axis=1).mean()

        return

    @property
    def results(self) -> pd.DataFrame:
        # TODO Create dataframe with explicit zeros
        precision = pd.DataFrame(dict(zip(self.col_names, scipy.sparse.find(self.scores_))))

        return precision
