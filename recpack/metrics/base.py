import logging
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.base import BaseEstimator


logger = logging.getLogger("recpack")


class Metric:
    def __init__(self):
        self.num_users_ = 0
        self.num_items_ = 0

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
        return self.value_

    @property
    def num_items(self) -> int:
        return self.num_items_

    @property
    def num_users(self) -> int:
        return self.num_users_

    @property
    def indices(self) -> Tuple[np.array, np.array]:
        row, col = np.indices((self.num_users_, self.num_items_))

        return row.flatten(), col.flatten()

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
            self.num_users_, self.num_items_ = y_true.shape

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

        self.user_id_map_ = np.array(nonzero_users)

        return y_true[nonzero_users, :], y_pred[nonzero_users, :]

    def map_users(self, users):
        if hasattr(self, "user_id_map_"):
            return self.user_id_map_[users]
        else:
            return users


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

    def get_top_K_ranks(self, y_pred: csr_matrix) -> csr_matrix:
        """
        Return csr_matrix of top K item ranks for every user.

        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: Sparse matrix containing ranks of top K predictions.
        :rtype: csr_matrix
        """
        U, I, V = [], [], []
        for row_ix, (le, ri) in enumerate(
                zip(y_pred.indptr[:-1], y_pred.indptr[1:])):
            K_row_pick = min(self.K, ri - le)

            if K_row_pick != 0:

                top_k_row = y_pred.indices[
                    le
                    + np.argpartition(y_pred.data[le:ri], list(range(-K_row_pick, 0)))[
                        -K_row_pick:
                    ]
                ]

                for rank, col_ix in enumerate(reversed(top_k_row)):
                    U.append(row_ix)
                    I.append(col_ix)
                    V.append(rank + 1)

        y_pred_top_K = csr_matrix((V, (U, I)), shape=y_pred.shape)

        self.y_pred_top_K_ = y_pred_top_K

        return y_pred_top_K

    @property
    def indices(self):
        row, col = self.y_pred_top_K_.nonzero()
        return row, col


class ElementwiseMetricK(MetricTopK):
    """
    Base class for all metrics that can be calculated for
    each user-item pair.

    Examples are: DCG, HR
    """

    @property
    def col_names(self):
        return ["user_id", "item_id", "score"]

    @property
    def results(self):
        scores = self.scores_.toarray()

        int_users, items = self.indices
        values = scores[int_users, items]

        users = self.map_users(int_users)

        return pd.DataFrame(dict(zip(self.col_names, (users, items, values))))

    @property
    def value(self):
        if hasattr(self, "value_"):
            return self.value_
        else:
            return self.scores_.sum(axis=1).mean()


class ListwiseMetricK(MetricTopK):
    """
    Base class for all metrics that can only be calculated
    at the list-level, i.e. one value for each user.

    Examples are: Diversity, nDCG, RR, Recall
    """

    @property
    def col_names(self):
        return ["user_id", "score"]

    @property
    def indices(self):
        row = np.arange(self.y_pred_top_K_.shape[0])
        col = np.zeros(self.y_pred_top_K_.shape[0], dtype=np.int32)
        return row, col

    @property
    def results(self):
        scores = self.scores_.toarray()

        int_users, items = self.indices
        values = scores[int_users, items]

        users = self.map_users(int_users)

        return pd.DataFrame(dict(zip(self.col_names, (users, values))))

    @property
    def value(self):
        if hasattr(self, "value_"):
            return self.value_
        else:
            return self.scores_.mean()


class GlobalMetricK(MetricTopK):
    """
    Base class for all metrics that can only be calculated
    as a global number across all items and users.

    Examples are: Coverage.
    """

    @property
    def results(self):

        return pd.DataFrame({"score": self.value})


class FittedMetric(Metric, BaseEstimator):
    """
    Base class for all metrics that need to be fit on a training set
    before they can be used.
    """

    def fit(self, X: csr_matrix):
        pass
