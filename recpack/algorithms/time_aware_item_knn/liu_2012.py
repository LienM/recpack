# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN
from recpack.algorithms.time_aware_item_knn.decay_functions import DecayFunction
from recpack.matrix import InteractionMatrix


class LiuDecay(DecayFunction):
    """Computes a logarithmic decay function.

    Every x in the `time_array` is discounted to:
    .. math::

        f(x) = log_\\alpha ((\\alpha-1)x + 1) + 1

    Where :math:`\\alpha` is the decay parameter.

    :param decay: The decay parameter, should be in the ]1, inf[ interval.
    :type decay: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        if not decay > 1:
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]1, inf [")

    def __init__(self, decay: float):
        self.validate_decay(decay)
        self.decay = decay

    def __call__(self, time_array: ArrayLike) -> ArrayLike:
        """Apply decay.
        :param time_array: array of time based weights, which will be decayed.
        Values should be in the [0, 1] interval.
        :type time_array: np.array
        """
        return (np.log(((self.decay - 1) * time_array) + 1) / np.log(self.decay)) + 1


class TARSItemKNNLiu2012(TARSItemKNN):
    """Time aware variant of ItemKNN which uses a logarithmic decay function.

    Algorithm as described in
    Y. Liu, Z. Xu, B. Shi and B. Zhang,
    "Time-Based K-nearest Neighbor Collaborative Filtering,"
    2012 IEEE 12th International Conference on Computer and Information Technology,
    Chengdu, China, 2012, pp. 1061-1065, doi: 10.1109/CIT.2012.217.

    Weights are computed based on their position in a user's history.
    The first item visited by a user gets value 1,
    last item visited gets value 2.
    The decay follows a logarithmic function:

    .. math::

        \\Gamma(x) = \\log_\\alpha ((\\alpha-1)x + 1) + 1

    where :math:`\\alpha` is the decay scaling parameter,
    and x is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int
    :param decay: The parameter of the logarithmic decay function. Defaults to 2.
    :type decay: float
    """

    DECAY_FUNCTIONS = {"liu": LiuDecay}

    def __init__(self, K: int = 200, decay: float = 2.0):
        super().__init__(K=K, fit_decay=decay, predict_decay=decay, decay_function="liu", similarity="cosine")
        self.decay = decay

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix, decay: float) -> csr_matrix:
        """Weight the interaction matrix based on age of the events.

        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :param decay: decay parameter, is 1/half_life
        :type decay: float
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """
        timestamp_mat = X.last_timestamps_matrix

        first_user_interactions = X.binary_values.multiply(self._compute_users_first_interaction(X))
        last_user_interactions = X.binary_values.multiply(timestamp_mat.max(axis=1))
        # the input for the decay is (t - t0_u) / tl_u
        # Where t0_u is the users first interaction and tl_u the last
        timestamp_mat.data = LiuDecay(self.decay)(
            (timestamp_mat.data - first_user_interactions.data) / last_user_interactions.data
        )
        return csr_matrix(timestamp_mat)

    def _compute_users_first_interaction(self, X: InteractionMatrix) -> np.array:
        """Compute the launch time of each item as the first time it was interacted with.

        If an item is not present in the dataset, their launch time is assumed 0

        :param X: InteractionMatrix to use for computation of launch times.
        :type X: InteractionMatrix
        :return: U x 1 array with the launch times of item i at index i.
        :rtype: np.array
        """
        first_interactions = X.timestamps.groupby(X.USER_IX).min()

        first_interactions_arr = np.zeros((X.shape[0], 1))
        first_interactions_arr[first_interactions.index, 0] = first_interactions
        return first_interactions_arr
