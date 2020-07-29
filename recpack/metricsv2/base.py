import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.base import BaseEstimator

import logging

from recpack.data.data_matrix import DataM

logger = logging.getLogger("recpack")

class Metric:

    @property
    def name(self):
        return str(self.__class__).lower()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        pass

    @property
    def results(self) -> pd.DataFrame:
        pass

    @property
    def value(self) -> float:
        pass


class ElementwiseMetric(Metric):
    """
    Base class for all metrics that can be calculated for
    each user-item pair.

    Examples are: DCG, HR
    """
    pass


class ListwiseMetric(Metric):
    """
    Base class for all metrics that can only be calculated
    at the list-level, i.e. one value for each user.

    Examples are: Diversity, nDCG, RR, Recall
    """
    pass


class GlobalMetric(Metric):
    """
    Base class for all metrics that can only be calculated
    as a global number across all items and users.

    Examples are: Coverage.
    """
    pass


class MetricTopK(Metric):
    """
    Base class for any metric computed on the TopK items for a user.
    """
    def __init__(self, K):
        super().__init__()
        self.K = K

    @property
    def name(self):
        return f"{super().name}_{self.K}"

    def get_topK(self, y_pred: csr_matrix) -> csr_matrix:
        # Get nonzero users
        nonzero_users = sorted(list(set(y_pred.nonzero()[0])))
        X = y_pred[nonzero_users, :].toarray()

        items = np.argpartition(X, -self.K)[:, -self.K:]

        U, I, V = [], [], []

        for ix, user in enumerate(nonzero_users):
            U.extend([user] * self.K)
            I.extend(items[ix])
            V.extend(X[ix, items[ix]])

        y_pred_top_K = csr_matrix(
            (V, (U, I)), dtype=y_pred.dtype, shape=y_pred.shape
        )

        return y_pred_top_K

class FittedMetric(Metric, BaseEstimator):
    """
    Base class for all metrics that need to be fit on a training set
    before they can be used.
    """
    def fit(self, X: csr_matrix):
        pass
