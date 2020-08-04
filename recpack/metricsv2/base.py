from typing import Tuple

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

        self.col_names = ["user_id", "item_id", "score"]

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
        """
        Make sure the dimensions of y_true and y_pred match.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :raises AssertionError: Shape mismatch between y_true and y_pred.
        :return: True if dimensions match.
        :rtype: bool
        """
        check = y_true.shape == y_pred.shape

        if not check:
            raise AssertionError(
                f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}"
            )
        else:
            # TODO Maybe this should be a separate method?
            self._num_users, self._num_items = y_true.shape

        return check

    def eliminate_empty_users(
        self, y_true: csr_matrix, y_pred: csr_matrix
    ) -> Tuple[csr_matrix, csr_matrix]:
        """
        Eliminate users that have no interactions, so
        no prediction could ever be right.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: (y_true, y_pred), with zero users eliminated.
        :rtype: Tuple[csr_matrix, csr_matrix]
        """
        nonzero_users = list(set(y_true.nonzero()[0]))

        return y_true[nonzero_users, :], y_pred[nonzero_users, :]


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

    # TODO Verify if this should be a Mixin
    def __init__(self, K):
        super().__init__()
        self.K = K

    @property
    def name(self):
        return f"{super().name}_{self.K}"

    def get_top_K_ranks(self, y_pred: csr_matrix) -> csr_matrix:
        """
        Return csr_matrix of top K item ranks for every user.

        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: Sparse matrix containing ranks of top K predictions.
        :rtype: csr_matrix
        """
        U, I, V = [], [], []
        for row_ix, (le, ri) in enumerate(zip(y_pred.indptr[:-1], y_pred.indptr[1:])):
            K_row_pick = min(self.K, ri - le)
            top_k_row = y_pred.indices[
                le + np.argpartition(y_pred.data[le:ri], -K_row_pick)[-K_row_pick:]
            ]

            for rank, col_ix in enumerate(reversed(top_k_row)):
                U.append(row_ix)
                I.append(col_ix)
                V.append(rank + 1)

        return csr_matrix((V, (U, I)), shape=y_pred.shape)


class FittedMetric(Metric, BaseEstimator):
    """
    Base class for all metrics that need to be fit on a training set
    before they can be used.
    """

    def fit(self, X: csr_matrix):
        pass
