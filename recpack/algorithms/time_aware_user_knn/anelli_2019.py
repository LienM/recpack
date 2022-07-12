import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms import Algorithm
from recpack.algorithms.time_aware_item_knn.decay_functions import exponential_decay
from recpack.matrix import InteractionMatrix, Matrix

# TODO: make memory efficient (if at all possible)


class TARSUserKNNAnelli(Algorithm):
    def __init__(self, decay, min_number_of_recommendations=20):
        self.decay = decay
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
        """Computes precursors for each user.

        Args:
            X (csr_matrix): _description_
        """
        if X.shape[1] < self.min_number_of_recommendations:
            raise ValueError(
                "Can't fit model on an interaction matrix with fewer items than "
                "the requested min_number_of_recommendations"
            )
        self.history_ = X.copy()

        # Will be used as fallback in prediction
        self.popularity_ = X.binary_values.sum(axis=0)

        last_t = X.last_timestamps_matrix

        candidate_precursor_counts = lil_matrix((X.shape[0], X.shape[0]))
        for user, hist in X.sorted_item_history:
            for item in hist:
                ti = last_t[user, item]
                # All users with an event on the item before this user did
                # The .multiply acts as an and operator on the two boolean matrices
                candidate_precursors = (last_t[:, item] != 0).multiply((last_t[:, item] < ti)).nonzero()[0]

                # Increment their cooc counts
                candidate_precursor_counts[user, candidate_precursors] = (
                    candidate_precursor_counts[user, candidate_precursors].toarray() + 1
                )

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
        user_input.data = exponential_decay(user_input.data, self.decay)

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
