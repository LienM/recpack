import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN
from recpack.matrix import InteractionMatrix


def liu_decay(time_array: np.array, alpha: float) -> np.array:
    """Computes a log based alpha function.

    .. math::

        log_\\alpha ((\\alpha-1)x + 1) + 1

    Where alpha is the alpha parameter, computed for each x in the time_array.

    :param time_array: array of time based weights, which will be decayed.
        Values should be in the [0, 1] interval.
    :type time_array: np.array
    :param alpha: The decay parameter, should be in the ]1, inf[ interval.
    :type alpha: float
    :returns: The decayed input time array. Larger values in the original array still have the highest values.
    :rtype: np.array

    """
    if not alpha > 1:
        raise ValueError(f"decay parameter alpha = {alpha} is not in the supported range: ]1, inf [")
    return (np.log(((alpha - 1) * time_array) + 1) / np.log(alpha)) + 1


class TARSItemKNNLiu2012(TARSItemKNN):
    """Time aware variant of ItemKNN which uses a logarithmic decay function.

    Algorithm as defined in Liu, Yue, et al. "Time-based k-nearest neighbor collaborative filtering."
    2012 IEEE 12th International Conference on Computer and Information Technology

    Weights are computed based on their position in a user's history. 1st item visited by a user gets value 1,
    last item visited gets value 2. The decay follows a logarithmic function.

    Uses the :class:`recpack.algorithms.time_aware_item_knn.decay_functions.LogDecay` decay function and cosine similarity.

    :param K: The number of neighbours to keep per item.
    :type K: int
    :param decay: The parameter of the logarithmic decay function. Defaults to 2.
    """

    def __init__(self, K=200, decay=2):
        super().__init__(K=K, fit_decay=decay, predict_decay=decay, decay_function="log", similarity="cosine")
        self.decay = decay

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix) -> csr_matrix:
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
        timestamp_mat.data = liu_decay(
            (timestamp_mat.data - first_user_interactions.data) / last_user_interactions.data, self.decay
        )
        return csr_matrix(timestamp_mat)

    def _add_decay_to_fit_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X)

    def _add_decay_to_predict_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X)

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
