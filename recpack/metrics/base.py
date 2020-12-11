import logging
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.base import BaseEstimator

from recpack.util import get_top_K_ranks

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

    # def get_top_K_ranks(self, y_pred: csr_matrix) -> csr_matrix:
    #     """
    #     Return csr_matrix of top K item ranks for every user.
    #
    #     :param y_pred: Predicted affinity of users for items.
    #     :type y_pred: csr_matrix
    #     :type use_rank: bool
    #     :return: Sparse matrix containing ranks of top K predictions.
    #     :rtype: csr_matrix
    #     """
    #     y_pred_top_K = get_top_K_ranks(y_pred)
    #     self.y_pred_top_K_ = y_pred_top_K
    #     return y_pred_top_K

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
    def results(self) -> pd.DataFrame:
        """Get the results for this metric.

        If there is a user with 0 recommendations,
        the output dataframe will contain K rows for
        that user, with item NaN and score 0

        :return: The results dataframe. With columns user, item, score
        :rtype: pd.DataFrame
        """
        scores = self.scores_.toarray()

        all_users = set(range(self.scores_.shape[0]))
        int_users, items = self.indices
        values = scores[int_users, items]

        # For all users in all_users but not in int_users,
        # add K items np.nan with value = 0
        missing_users = all_users.difference(set(int_users))

        # This should barely occur, so it's not too bad to append np arrays.
        for u in list(missing_users):
            for i in range(self.K):
                int_users = np.append(int_users, u)
                values = np.append(values, 0)
                items = np.append(items, np.nan)

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
