# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN
from recpack.matrix import InteractionMatrix


class TARSItemKNNLee(TARSItemKNN):
    """Time aware variant of ItemKNN which uses a hard-coded decay matrix and cosine or pearson similarity. 

    Algorithm implementing
    Tong Queue Lee, Young Park, Yong-Tae Park,
    A time-based approach to effective recommender systems using implicit feedback,
    Expert Systems with Applications,
    Volume 34, Issue 4,
    2008,
    Pages 3055-3062,
    ISSN 0957-4174,
    https://doi.org/10.1016/j.eswa.2007.06.031.

    Weights are used both for prediction and for training.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param w: Shape of the weighting matrix, defaults to 5.
        ``[3, 5]`` are supported.
    :type w: int, optional
    :param similarity: Which similarity measure to use. Defaults to ``"cosine"``.
        ``["cosine", "pearson"]`` are supported.
    :type similarity: str, optional
    """

    SUPPORTED_SIMILARITIES = ["cosine", "pearson"]
    W_MAP = {
        3: np.array(
            [
                [0.7, 1.7, 2.7],
                [1.0, 2.0, 3.0],
                [1.3, 2.3, 3.3],
            ]
        ),
        5: np.array(
            [
                [0.2, 1.2, 2.2, 3.2, 4.2],
                [0.4, 1.4, 2.4, 3.4, 4.4],
                [0.6, 1.6, 2.6, 3.6, 4.6],
                [0.8, 1.8, 2.8, 3.8, 4.8],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        ),
    }

    def __init__(self, K: int = 200, w: int = 5, similarity: str = "cosine"):
        super().__init__(K, similarity=similarity, fit_decay=0, predict_decay=0)

        if w not in self.W_MAP:
            raise ValueError(f"Weight matrix {w} is not supported. Only 3 and 5 are allowed.")

        self.w = w
        self.W = self.W_MAP[w]

    def _add_decay_to_fit_matrix(self, X: InteractionMatrix) -> csr_matrix:
        """Add decay to each user, item interaction based on the launch time of the item,
        and the last time the user interacted with the item.

        Weights are defined in the weight_matrix property.

        :param X: InteractionMatrix with events to use to generate a weighted matrix.
        :type X: InteractionMatrix
        :return: Weighted user x item matrix. At position u, i the weight of user u interacting with item i is stored.
        :rtype: csr_matrix
        """
        launch_times = self._compute_launch_times(X)
        launch_width = launch_times.max() - launch_times.min()
        launch_min = launch_times.min()
        launch_window_width = launch_width / self.w
        # Create division points
        launch_splits = [np.ceil(launch_min + i * launch_window_width) for i in range(1, self.w + 1)]

        timestamps_mat = X.last_timestamps_matrix
        timestamps_min = X.timestamps.min()
        timestamps_max = X.timestamps.max()
        timestamps_width = timestamps_max - timestamps_min
        timestamps_window_width = timestamps_width / self.w

        timestamps_splits = [np.ceil(timestamps_min + i * timestamps_window_width) for i in range(1, self.w + 1)]

        def get_weight_index(arr, value):
            """Get the index of the first value in the array that is greater than or equal to value"""
            return next(ix for ix, val in enumerate(arr) if val >= value)

        X = dok_matrix(X.shape)
        for user, item in zip(*timestamps_mat.nonzero()):
            ts = timestamps_mat[user, item]
            lt = launch_times[item]
            launch_ix = get_weight_index(launch_splits, lt)
            timestamps_ix = get_weight_index(timestamps_splits, ts)
            w = self.W[launch_ix, timestamps_ix]
            X[user, item] = w

        return X.tocsr()

    def _compute_launch_times(self, X: InteractionMatrix) -> np.array:
        """Compute the launch time of each item as the first time it was interacted with.

        If an item is not present in the dataset, their launch time is assumed 0

        :param X: InteractionMatrix to use for computation of launch times.
        :type X: InteractionMatrix
        :return: 1D array with the launch times of item i at index i.
        :rtype: np.array
        """
        launch_times = X.timestamps.groupby(X.ITEM_IX).min()

        launch_times_arr = np.zeros(X.shape[1])
        launch_times_arr[launch_times.index] = launch_times
        return launch_times_arr
