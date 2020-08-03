import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from sklearn.base import BaseEstimator

import logging

from recpack.data.data_matrix import DataM

logger = logging.getLogger("recpack")


class Metric:

    def __init__(self):
        super().__init__()
        self._num_users = 0
        self._num_items = 0
        self._value = 0

    @property
    def name(self):
        return str(self.__class__).lower()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """
        Calculates this Metric for all users.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        pass

    @property
    def results(self) -> pd.DataFrame:
        pass

    @property
    def value(self) -> float:
        return self._value

    @property
    def num_items(self) -> int:
        return self._num_items

    @property
    def num_users(self) -> int:
        return self._num_users

    def verify_shape(self, y_true: csr_matrix, y_pred: csr_matrix) -> bool:
        check = y_true.shape == y_pred.shape

        if not check:
            raise AssertionError(f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}")
        else:
            # TODO Maybe this should be a separate method?
            self._num_users, self._num_items = y_true.shape

        return True


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
    #TODO Verify if this should be a Mixin
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

        items = np.argpartition(X, -self.K)[:, -self.K :]

        U, I, V = [], [], []

        for ix, user in enumerate(nonzero_users):
            U.extend([user] * self.K)
            I.extend(items[ix])
            V.extend(X[ix, items[ix]])

        y_pred_top_K = csr_matrix((V, (U, I)), dtype=y_pred.dtype, shape=y_pred.shape)

        return y_pred_top_K


class FittedMetric(Metric, BaseEstimator):
    """
    Base class for all metrics that need to be fit on a training set
    before they can be used.
    """

    def fit(self, X: csr_matrix):
        pass
