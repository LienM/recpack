import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms import Algorithm
from recpack.algorithms.time_aware_item_knn.decay_functions import ExponentialDecay
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values, to_binary

# TODO: make memory efficient (if at all possible)


class TARSUserKNNAnelli(Algorithm):
    """Time aware user KNN based on precursor users.

    Described in Anelli, Vito Walter, et al. "Local popularity and time in top-n recommendation."
    European Conference on Information Retrieval. Springer, Cham, 2019.

    For each user a set of precursor users are computed.
    These are users that frequently interact with items before the user interacts with those items.
    These precursor users are assumed to inspire the user,
    and so the unseen items they interacted with are what will be recommended.

    For recommendation an exponential decay :math:`e^{\\lambda \\Delta t}` is applied to give more weight
    to recently active precursors as well as recently visited items by precursors.
    Given that :math:`t_0` is the 'now' timestamp
    (computed as the last interaction in the prediction input matrix + 1 second),
    :math:`t_{u',l}` is the last interaction of the precursor :math:`u'`
    and :math:`t_{u',i}` is the time of interaction between the precursor and an item.

    .. math::

        \\Delta t = |t_0 - 2 t_{u',l} + t_{u',i}|

    Users with no precursors get global popularity.

    .. warning::

        In contrast to other algoritms in RecPack, this algorithm uses popularity as fallback.
        When the model can not give personalised recommendations to the user.

        This can give it an unfair advantage in situations where most algorithms
        can not personalise for a significant portion of the user base.

    :param decay: The scaling factor :math:`\\lambda` in the exponential.
    :type decay: float
    :param min_number_of_recommendations: The minimal requested number of recommendations per user, if users get
        recommended fewer items, their recommendations are filled up with popularity.
        Defaults to 100
    :type min_number_of_recommendations: int, optional
    """

    def __init__(self, decay: float, min_number_of_recommendations: int = 100):
        self.decay = decay
        self.decay_func = ExponentialDecay(self.decay)
        self.min_number_of_recommendations = min_number_of_recommendations

    def _transform_fit_input(self, X: Matrix):
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix):
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: InteractionMatrix):
        """Computes precursors for each user."""
        if X.shape[1] < self.min_number_of_recommendations:
            raise ValueError(
                "Can't fit model on an interaction matrix with fewer items than "
                "the requested min_number_of_recommendations"
            )
        self.history_ = X.copy()

        # Will be used as fallback in prediction
        self.popularity_ = get_top_K_values(
            csr_matrix(X.binary_values.sum(axis=0)), self.min_number_of_recommendations
        )

        last_t = X.last_timestamps_matrix

        candidate_precursor_counts = lil_matrix((X.shape[0], X.shape[0]))
        for user in X.active_users:
            ts_cooc = last_t.multiply(to_binary(last_t[user, :]))
            ts_user = to_binary(ts_cooc).multiply(last_t[user, :])
            candidate_precursor_counts[user] = (ts_cooc < ts_user).sum(axis=1)

        # precursors are all candidate precursors whose number of cooc are bigger than the avg per user
        avg_cooc_per_user = candidate_precursor_counts.mean(axis=1)
        self.precursors_ = csr_matrix(candidate_precursor_counts > avg_cooc_per_user, dtype=int)

    def _predict(self, X: InteractionMatrix) -> csr_matrix:
        now = X.timestamps.max() + 1  # 1 second past the last timestamp in the history matrix.

        user_last = X.last_timestamps_matrix.max(axis=1)

        # matrix with delta t on each u,i position
        # Where delta t is |t0 - 2t_ul + t_ui|
        user_input = np.abs(
            (self.history_.binary_values > 0).multiply(now - 2 * user_last.toarray())
            + self.history_.last_timestamps_matrix
        )
        user_input.data = self.decay_func(user_input.data)

        # fallback popularity for active users with no precursors
        predictions = lil_matrix(self.precursors_ @ user_input)
        nonpredicted_users = X.active_users - set(predictions.nonzero()[0])
        predictions[list(nonpredicted_users)] = self.popularity_

        # Users with too few predictions get filled up with popularity
        # For this we scale the popularity scores, such that they all fall below the minimal personalised score.
        # To avoid adding them to all items though, we subtract popularity scores from already recommended items.
        incomplete_users = ((predictions > 0).sum(axis=1) < self.min_number_of_recommendations).nonzero()[0]
        max_pop = self.popularity_.max()
        for user in incomplete_users:
            scaling_factor = min(predictions[user].data[0]) / (max_pop + 1)
            predictions[user] = (
                predictions[user]
                + scaling_factor * self.popularity_
                - (predictions[user] > 0).multiply(scaling_factor * self.popularity_)
            )
        return csr_matrix(predictions)
