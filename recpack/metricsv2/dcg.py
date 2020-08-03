import logging

import scipy.sparse
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


from recpack.metricsv2.base import ElementwiseMetric, ListwiseMetric, MetricTopK
from recpack.metricsv2.util import sparse_divide_nonzero


logger = logging.getLogger("recpack")


class DCGK(ElementwiseMetric, MetricTopK):
    def __init__(self, K):
        MetricTopK.__init__(self, K)

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        self.verify_shape(y_true, y_pred)

        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

        denominator = y_pred_top_K.multiply(y_true)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self.dcg_ = dcg

        self._value = self.dcg_.sum(axis=1).mean()

        return

    @property
    def results(self):
        # TODO Create dataframe with explicit zeros
        hits = pd.DataFrame(dict(zip(self.col_names, scipy.sparse.find(self.scores_))))

        return hits

# TODO implement NDCG
# ? does it start from DCG?
#   -> Problem is that it is then both an elementwise and a listwise metric
# Maybe give it a DCG member? and then IDCG template func,
# so that we can get the results and normalize them?
# At that point it's almost easier to just implement it.
class NDCG(ListwiseMetric):
    pass
