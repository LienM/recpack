import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN
from recpack.data.matrix import InteractionMatrix


class TARSItemKNNLee(TARSItemKNN):
    SUPPORTED_SIMILARITIES = ["cosine", "pearson"]

    def __init__(self, K: int = 200, num_windows: int = 3, similarity: str = "cosine"):
        super().__init__(K, similarity=similarity, fit_decay=0, predict_decay=0)
        self.num_windows = num_windows

    @property
    def weight_matrix(self):
        """Matrix with weights, row indices based on age of launch time of an item, column indices based on age of the event

        Top left is old/old, bottom right is new/new.
        Position 1,1 will have lower weight than both 1,2, 2,1 and 2,2.
        """
        stepsize = 1 / self.num_windows
        return np.array(
            [[i + (j * stepsize) for j in range(1, self.num_windows + 1)] for i in range(self.num_windows)]
        ).T

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix, decay=None) -> csr_matrix:
        """Add decay to each user, item interaction based on the launch time of the item,
        and the last time the user interacted with the item.

        Weights are defined in the weight_matrix property.

        :param X: InteractionMatrix with events to use to generate a weighted matrix.
        :type X: InteractionMatrix
        :param decay: Unused decay scaling parameter
        :type decay: Optional[float]
        :return: Weighted user x item matrix. At position u, i the weight of user u interacting with item i is stored.
        :rtype: csr_matrix
        """
        launch_times = self._compute_launch_times(X)
        launch_width = launch_times.max() - launch_times.min()
        launch_min = launch_times.min()
        launch_window_width = launch_width / self.num_windows
        # Create division points
        launch_splits = [np.ceil(launch_min + i * launch_window_width) for i in range(1, self.num_windows + 1)]

        timestamps_mat = X.last_timestamps_matrix
        timestamps_min = X.timestamps.min()
        timestamps_max = X.timestamps.max()
        timestamps_width = timestamps_max - timestamps_min
        timestamps_window_width = timestamps_width / self.num_windows

        timestamps_splits = [
            np.ceil(timestamps_min + i * timestamps_window_width) for i in range(1, self.num_windows + 1)
        ]

        def get_weight_index(arr, value):
            """Get the index of the first value in the array that is greater than or equal to value"""
            return next(ix for ix, val in enumerate(arr) if val >= value)

        X = lil_matrix(X.shape)
        for user, item in zip(*timestamps_mat.nonzero()):
            ts = timestamps_mat[user, item]
            lt = launch_times[item]
            launch_ix = get_weight_index(launch_splits, lt)
            timestamps_ix = get_weight_index(timestamps_splits, ts)
            w = self.weight_matrix[launch_ix, timestamps_ix]
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


class TARSItemKNNLee_W3(TARSItemKNNLee):
    def __init__(self, K: int = 200, similarity: str = "cosine"):
        super().__init__(K, num_windows=3, similarity=similarity)

    # fmt: off
    weight_matrix = np.array([
        [0.7, 1.7, 2.7],
        [1.0, 2.0, 3.0],
        [1.3, 2.3, 3.3],
    ])
    # fmt: on


class TARSItemKNNLee_W5(TARSItemKNNLee):
    def __init__(self, K: int = 200, similarity: str = "cosine"):
        super().__init__(K, num_windows=5, similarity=similarity)
