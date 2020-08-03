import logging

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metricsv2.base import ElementwiseMetric, MetricTopK


logger = logging.getLogger("recpack")


class RecallK(ElementwiseMetric, MetricTopK):
    def __init__(self, K):
        ElementwiseMetric.__init__(self)
        MetricTopK.__init__(self, K)

        self.col_names = ["user_id", "item_id", "score"]

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        self.verify_shape(y_true, y_pred)

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)

        y_pred_top_K = self.get_top_K(y_pred)

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        self.scores_ = scores.tocsr()

        self.y_true_ = y_true

        self._value = (self.scores.sum(axis=1) / self.K).mean()

        return

    @property
    def results(self) -> pd.DataFrame:
        # TODO Create dataframe with explicit zeros
        hits = pd.DataFrame(dict(zip(self.col_names, scipy.sparse.find(self.scores_))))

        return hits
