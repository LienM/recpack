# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms import Algorithm
from recpack.algorithms.util import get_batches
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_ranks, get_top_K_values


class STAN(Algorithm):
    """Sequence and Time Aware Neighbourhoods algorithm.

    Algorithm presented by Garg, Diksha, et al.
    "Sequence and time aware neighborhood for session-based recommendations: STAN."

    The algorithm is a modified version of UserKNN with several decay schemes applied.

    Each of the user's interactions are weighted by multiplying them with

    .. math::

        e^{- \\lambda_1 \\, (t_{max} - t_i)}

    Where lambda_1 is the ``interaction_decay`` parameter.

    A second weighting scheme is applied when computing session similarities.
    The time of a session is the last timestamp in that session.

    Session similarities are weighted by multiplication with

    .. math::

        e^{- \\lambda_2 \\, |T_{s1} - T_{s_2}|}

    Where lambda_2 is the ``session_decay`` parameter.

    A final weighting is applied to recommend items closest to
    the last matching item between similar users.
    For each item in a neighbours history,
    the session similarity is weighted additionally by

    .. math::

        e^{- \\lambda_3 \\, |pos_i - pos_{matching}|}

    Where lambda_3 is the ``distance_from_match_decay`` parameter.

    .. note::

        We modified the decay computations from the paper,
        by using a multiplicative weight, rather than a division.
        This allows us to easily disable a weight.
        Typical values for decay parameters will be between 0 and 1.

    :param K: The amount of sessions to be considered as neighbourhood,
        defaults to 200
    :type K: int, optional
    :param interaction_decay: The decay factor for session history weighting,
        defaults to 1/3600
    :type interaction_decay: float, optional
    :param session_decay: The decay factor for session similarity computation,
        defaults to 1/3600
    :type session_decay: float, optional
    :param distance_from_match_decay: The decay factor for
        prediction item weighting, defaults to 1
    :type distance_from_match_decay: float, optional

    """

    def __init__(
        self,
        K: int = 200,
        interaction_decay: float = 1 / 3600,
        session_decay: float = 1 / 3600,
        distance_from_match_decay: float = 1,
    ):
        super().__init__()
        self.K = K
        self.interaction_decay = interaction_decay
        self.session_decay = session_decay
        self.distance_from_match_decay = distance_from_match_decay

    def _transform_fit_input(self, X: Matrix) -> InteractionMatrix:
        """

        :param X: User-item interaction matrix to fit the model to
        :type X: Matrix
        :return:
        :rtype:
        """
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> InteractionMatrix:
        """

        :param X: User-item interaction matrix used as input to predict
        :type X: Matrix
        :return:
        :rtype:
        """
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: InteractionMatrix) -> None:
        # STAN is not a model based model, but we can precompute some values
        self.sessions_ = X
        session_interactions_timestamps = X.last_timestamps_matrix
        self.session_interactions_positions_ = timestamp_matrix_to_position(session_interactions_timestamps)
        self.historical_session_timestamps_ = session_interactions_timestamps.max(axis=1)  # |U| x 1 matrix

    def _predict(self, X: InteractionMatrix) -> csr_matrix:
        timestamp_matrix = X.last_timestamps_matrix

        # Construct the session similarity matrix & apply weighting and topK
        # Do this iteratively for sections of users
        full_session_similarity_matrix = lil_matrix((X.shape[0], X.shape[0]))

        for user_batch in get_batches(X.active_users, batch_size=1000):
            session_similarity = self._compute_session_similarity(timestamp_matrix[user_batch, :]).tolil()
            session_similarity = session_similarity.multiply(
                self._compute_session_similarity_weights(timestamp_matrix[user_batch, :], session_similarity)
            )
            # Remove self similarity
            # Rows are indexed 0 - len(batch), cols are index by original user ids.
            session_similarity[np.arange(len(user_batch)), user_batch] = 0

            full_session_similarity_matrix[user_batch, :] = get_top_K_values(session_similarity.tocsr(), K=self.K)
        predictions = self._compute_prediction_scores(full_session_similarity_matrix, X)
        return predictions

    def _compute_session_similarity(self, session_timestamps: csr_matrix) -> csr_matrix:
        """Computes session similarity given the timestamps of session interactions.

        :param session_timestamps: Matrix with timestamps of interactions
        :type session_timestamps: csr_matrix
        :return: A |session| x |session| matrix with similarities.
            2nd dimension are the training sessions.
        :rtype: csr_matrix
        """

        # Compute the per item weights for each of the sessions:
        # w1(i,s) = exp((p(i,s) - l(s)) * interaction_decay)
        # Where p(i,s) is the position of item i in session s
        # and l(s) is the length of the session

        # We first get the reverse positions
        # (when you would start counting from the last event),
        # which are equivalent to (l(s) - p(i,s)) + 1
        # So our computation changes to exp(-((rank(i,s) - 1) * interaction_decay))
        session_ranks = get_top_K_ranks(session_timestamps)
        weighted_sessions = session_ranks.copy()
        weighted_sessions.data = np.exp(-(weighted_sessions.data - 1) * self.interaction_decay)

        # Compute  similarity between the weighted sessions, and the training sessions
        # Similarity computed between sessions a and b as
        #   <a, b> / sqrt(|a| * |b|)

        session_similarity = weighted_sessions @ self.sessions_.binary_values.T
        denominator_part_1 = session_ranks.max(axis=1)
        denominator_part_1.data = 1 / np.sqrt(denominator_part_1.data)
        denominator_part_2 = self.session_interactions_positions_.max(axis=1)
        denominator_part_2.data = 1 / np.sqrt(denominator_part_2.data)

        session_similarity = session_similarity.multiply(denominator_part_1).multiply(denominator_part_2.T)
        return session_similarity

    def _compute_session_similarity_weights(
        self, session_timestamps: csr_matrix, session_similarities: csr_matrix
    ) -> csr_matrix:
        """Session similarities will be weighted proportional
        to the time between the training and input sessions.

        w(s, s_j) = exp(-(t(s) - t(s_j))*session_decay)

        :param session_timestamps: [description]
        :type session_timestamps: [type]
        :param session_similarities: The similarities between sessions
        :type session_similarities: csr_matrix
        :return: [description]
        :rtype: csr_matrix
        """
        sessions_last_timestamp = session_timestamps.max(axis=1)

        # To compute - t(s) - t(s_j) i.e. t(s_j) - t(s)
        # only for those sessions that are actually similar,
        # we compute 2 sparse matrices with only values
        # for similar sessions (match at least 1 item)

        # 1st matrix contains the timestamp of the input sessions
        # in their respective rows.
        intersection_session_last_timestamps = (session_similarities > 0).multiply(sessions_last_timestamp)
        # 2nd matrix contains the timestamp of the training sessions
        # in their respective columns.
        intersection_historical_last_timestamps = (session_similarities > 0).multiply(
            self.historical_session_timestamps_.T
        )

        # By subtracting the two, we get at position (i,j) the value t(s_j) - t(s_i)
        # where j is a training session, and i is an input session
        session_similarity_weights = intersection_historical_last_timestamps - intersection_session_last_timestamps
        session_similarity_weights.data = np.exp(session_similarity_weights.data * self.session_decay)

        return session_similarity_weights

    def _compute_prediction_scores(self, session_similarity: csr_matrix, X: InteractionMatrix) -> csr_matrix:
        """Computes recommendation scores for active users given the session similarity matrix.

        :param session_similarity: The matrix of similarities between sessions.
            Dimension 1 = input sessions.
            Dimension 2 = training sessions.
        :type session_similarity: csr_matrix
        :param X: The input session interaction matrix.
        :type X: InteractionMatrix
        :return: |U| x |I| scores matrix.
        :rtype: csr_matrix
        """
        results = lil_matrix(X.shape)
        binary_history = X.binary_values
        for session in X.active_users:
            history = binary_history[session, :]
            # get the similarity between the session and training sessions.
            neighborhood_scores = session_similarity[session, :].toarray()

            # Get the positions of visits in the neighborhood sessions
            neighborhood_positions = lil_matrix(
                self.session_interactions_positions_.multiply((neighborhood_scores > 0).T)
            )

            # Find the last matching position with each neighbour session
            matching_positions = neighborhood_positions.multiply(history)
            last_match = matching_positions.max(axis=1)

            # Compute the weight for each item in the matching neighbours
            # The last matching neighbour gets a final weight of 0,
            # which is different than in the paper.
            # In the paper it gets a weight of 1 (exp(0)).
            # We make this decision to make the computation easier
            #   (the 0 distance disappears in the csr matrix)
            # During evaluation in the paper, the history items are also removed,
            # so wether it is 1 or 0 does not impact the reproduction results.
            # Because recpack does not always remove history items,
            # it makes more sense to not recommend this last matching item as well.
            item_weights = neighborhood_positions - (neighborhood_positions > 0).multiply(last_match.A)

            item_weights.data = np.exp(-np.abs(item_weights.data) * self.distance_from_match_decay)

            results[session] = neighborhood_scores @ item_weights

        return results.tocsr()


def timestamp_matrix_to_position(timestamp_matrix):
    """Returns a matrix of positions from 1 (first interaction) to l (last interaction).

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :return: Matrix with K values per row.
    :rtype: csr_matrix
    """
    U, I, V = [], [], []
    for row_ix, (le, ri) in enumerate(zip(timestamp_matrix.indptr[:-1], timestamp_matrix.indptr[1:])):
        row_pick = ri - le

        if row_pick != 0:
            for rank, sort_ix in enumerate(np.argsort(timestamp_matrix.data[le:ri])):
                U.append(row_ix)
                I.append(timestamp_matrix.indices[le + sort_ix])
                V.append(rank + 1)

    return csr_matrix((V, (U, I)), shape=timestamp_matrix.shape)
