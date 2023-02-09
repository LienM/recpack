# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.base import BaseEstimator

from recpack.util import get_top_K_ranks


logger = logging.getLogger("recpack")


class Metric:
    """Base class for all metrics.

    A Metric object is stateful, i.e. after ``calculate``
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`
    """

    def __init__(self):
        self.num_users_ = 0
        self.num_items_ = 0

    @property
    def name(self):
        """Name of the metric."""
        return self.__class__.__name__

    def _calculate(self, y_true, y_pred) -> None:
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Calculates this metric for all nonzero users in ``y_true``,
        given true labels and predicted scores.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        y_true, y_pred = self._eliminate_empty_users(y_true, y_pred)
        self._verify_shape(y_true, y_pred)
        self._set_shape(y_true)

        self._calculate(y_true, y_pred)

    @property
    def results(self):
        """Detailed results of the metric."""
        return pd.DataFrame({"score": [self.value]})

    @property
    def value(self) -> float:
        """The global metric value."""
        return self.value_

    @property
    def num_items(self) -> int:
        """Dimension of the item-space in both ``y_true`` and ``y_pred``"""
        return self.num_items_

    @property
    def num_users(self) -> int:
        """Dimension of the user-space in both ``y_true`` and ``y_pred``
        after elimination of users without interactions in ``y_true``.
        """
        return self.num_users_

    @property
    def _indices(self) -> Tuple[np.array, np.array]:
        """Indices in the prediction matrix for which scores were computed."""
        row, col = np.indices((self.num_users_, self.num_items_))

        return row.flatten(), col.flatten()

    def _verify_shape(self, y_true: csr_matrix, y_pred: csr_matrix) -> bool:
        """Make sure the dimensions of y_true and y_pred match.

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
            raise AssertionError(f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}")
        return check

    def _set_shape(self, y_true):
        self.num_users_, self.num_items_ = y_true.shape

    def _eliminate_empty_users(self, y_true: csr_matrix, y_pred: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """Eliminate users that have no interactions in ``y_true``.

        We cannot make accurate predictions of interactions for
        these users as there are none.

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

    def _map_users(self, users):
        """Map internal identifiers of users to actual user identifiers."""
        if hasattr(self, "user_id_map_"):
            return self.user_id_map_[users]
        else:
            return users


class MetricTopK(Metric):
    """Base class for all metrics computed based on the Top-K recommendations for every user.

    A MetricTopK object is stateful, i.e. after ``calculate``
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__()
        self.K = K

    @property
    def name(self):
        """Name of the metric."""
        return f"{super().name}_{self.K}"

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row, col = self.y_pred_top_K_.nonzero()
        return row, col

    def _calculate(self, y_true, y_pred_top_K):
        """Computes metric given true labels ``y_true`` and predicted scores ``y_pred``. Only Top-K recommendations are considered.

        To be implemented in the child class.

        :param y_true: Expected interactions per user.
        :type y_true: csr_matrix
        :param y_pred_top_K: Ranks for topK recommendations per user
        :type y_pred_top_K: csr_matrix
        """
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Computes metric given true labels ``y_true`` and predicted scores ``y_pred``. Only Top-K recommendations are considered.

        Detailed metric results can be retrieved with :attr:`results`.
        Global aggregate metric value is retrieved as :attr:`value`.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        # Perform checks and cleaning
        y_true, y_pred = self._eliminate_empty_users(y_true, y_pred)
        self._verify_shape(y_true, y_pred)
        self._set_shape(y_true)

        # Compute the topK for the predicted affinities
        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        # Compute the metric.
        self._calculate(y_true, y_pred_top_K)


class ElementwiseMetricK(MetricTopK):
    """Base class for all metrics that can be calculated for
    each user-item pair in the Top-K recommendations.

    :attr:`results` contains an entry for each user-item pair.

    Examples are: HitK, IPSHitRateK

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self):
        """The names of the columns in the results DataFrame."""
        return ["user_id", "item_id", "score"]

    @property
    def results(self) -> pd.DataFrame:
        """Get the detailed results for this metric.

        Contains an entry for every user-item pair in the Top-K recommendations list of every user.

        If there is a user with no recommendations,
        the results DataFrame will contain K rows for
        that user with item_id = NaN and score = 0.

        :return: The results DataFrame with columns: user_id, item_id, score
        :rtype: pd.DataFrame
        """
        scores = self.scores_.toarray()

        all_users = set(range(self.scores_.shape[0]))
        int_users, items = self._indices
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

        users = self._map_users(int_users)

        return pd.DataFrame(dict(zip(self.col_names, (users, items, values))))

    @property
    def value(self):
        """Global metric value obtained by summing up scores for every user then taking the average over all users."""
        return self.scores_.sum(axis=1).mean()


class ListwiseMetricK(MetricTopK):
    """Base class for all metrics that can be calculated for every Top-K recommendation list,
    i.e. one value for each user.
    Examples are: DiversityK, NDCGK, ReciprocalRankK, RecallK

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self):
        """The names of the columns in the results DataFrame."""
        return ["user_id", "score"]

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row = np.arange(self.y_pred_top_K_.shape[0])
        col = np.zeros(self.y_pred_top_K_.shape[0], dtype=np.int32)
        return row, col

    @property
    def results(self):
        """Get the detailed results for this metric.

        Contains an entry for every user.

        :return: The results DataFrame with columns: user_id, score
        :rtype: pd.DataFrame
        """
        scores = self.scores_.toarray()

        int_users, items = self._indices
        values = scores[int_users, items]

        users = self._map_users(int_users)

        return pd.DataFrame(dict(zip(self.col_names, (users, values))))

    @property
    def value(self):
        """Global metric value obtained by taking the average over all users."""
        return self.scores_.mean()


class GlobalMetricK(MetricTopK):
    """
    Base class for all metrics that can only be calculated
    as a global value across all items and users.

    Examples are: Coverage

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    pass


class FittedMetric(Metric, BaseEstimator):
    """
    Base class for all metrics that need to be fit on a training set
    before they can be used.

    Examples are: IntraListDiversityK, IPSHitRateK
    """

    def fit(self, X: csr_matrix):
        pass
