# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Module with time-dependent ItemKNN implementations"""

import numpy as np

from scipy.sparse import csr_matrix

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.nearest_neighbour import (
    compute_conditional_probability,
    compute_cosine_similarity,
    compute_pearson_similarity,
)
from recpack.algorithms.time_aware_item_knn.decay_functions import (
    exponential_decay,
    log_decay,
    linear_decay,
    linear_decay_steeper,
    concave_decay,
    convex_decay,
    inverse_decay,
)
from recpack.data.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values

EPSILON = 1e-13


class TARSItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """ItemKNN algorithm where older interactions have less weight during prediction, training or both.

    This class is the baseclass for ItemKNN weighting approaches, combining their functionality,
    and allowing unpublished combinations of settings.
    Includes work by Liu, Nathan N., et al. (2010), Ding et al. (2005) and Lee et al. (2007).

    The standard framework for all of these approaches can be summarised as:

    - When training the user interaction matrix is weighted to take into account temporal information available.
    - Similarities are computed on this weighted matrix, using various similarity measures.
    - When predicting the interactions are similarly weighted, giving more weight to more recent interactions.
    - Recommendation scores are obtained by multiplying the weighted interaction matrix with
      the previously computed similarity matrix.

    The default weighting in this base class is:
    .. math::

        e^{- \\alpha \\text{age}/\\text{decay_interval}}

    Where alpha is the decay scaling parameter,
    and age is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    :param K: Amount of neighbours to keep. Defaults to 200.
    :type K: int, Optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to (1/3600), such that the half life is 1 hour.
    :type fit_decay: float, Optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to (1/3600), such that the half life is 1 hour.
    :type predict_decay: float, Optional
    :param decay_interval: Interval in seconds to consider when computing decay.
        Allows better specifications of parameters for large scale datasets with ages in terms of days / months.
        Defaults to 1 (second).
    :type decay_interval: int, optional
    :param similarity: Which similarity measure to use. Defaults to `"cosine"`.
    :type similarity: str, Optional
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "pearson"]
    DECAY_FUNCTIONS = {
        "exponential": exponential_decay,
        "log": log_decay,
        "linear": linear_decay,
        "linear_steeper": linear_decay_steeper,
        "concave": concave_decay,
        "convex": convex_decay,
        "unity": lambda x: x,
        "inverse": inverse_decay,
    }

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
        decay_interval: int = 1,
        similarity: str = "cosine",
        decay_function: str = "exponential",
    ):
        # Uses other default parameters for ItemKNN
        super().__init__(K=K)
        self.fit_decay = fit_decay
        self.predict_decay = predict_decay

        if decay_interval <= 0 or type(decay_interval) == float:
            raise ValueError("Decay_interval needs to be a positive nonzero integer")

        self.decay_interval = decay_interval

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} is not supported")
        self.similarity = similarity

        if decay_function not in self.DECAY_FUNCTIONS:
            raise ValueError(f"decay function {decay_function} is not supported")

        self.decay_function = decay_function

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X.

        Scores are computed by matrix multiplication of weighted X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        X = self._add_decay_to_interaction_matrix(X, self.predict_decay)
        return super()._predict(X)

    def _transform_fit_input(self, X: Matrix) -> InteractionMatrix:
        """Weight each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> InteractionMatrix:
        """Weight each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item."""
        X = self._add_decay_to_interaction_matrix(X, self.fit_decay)

        if self.similarity == "cosine":
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X)
        elif self.similarity == "pearson":
            item_similarities = compute_pearson_similarity(X)

        item_similarities = get_top_K_values(item_similarities, self.K)

        self.similarity_matrix_ = item_similarities

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix, decay: float) -> csr_matrix:
        """Weight the interaction matrix based on age of the events.

        If decay is 0, it is assumed to be disabled, and so we just return binary matrix.
        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :param decay: decay parameter, is 1/half_life.
        :type decay: float
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """

        if decay == 0:
            return X.binary_values

        timestamp_mat = X.last_timestamps_matrix
        # The maximal timestamp in the matrix is used as 'now',
        # age is encoded as now - t
        now = timestamp_mat.data.max()
        timestamp_mat.data = self.DECAY_FUNCTIONS[self.decay_function](
            (now - timestamp_mat.data) / self.decay_interval, decay
        )
        return csr_matrix(timestamp_mat)


class TARSItemKNNCoocDistance(TARSItemKNN):

    SUPPORTED_SIMILARITIES = ["cooc", "conditional_probability", "hermann"]
    """Supported similarities, ``hermann`` is the similarity defined in Hermann et al. (2010), 
    dividing the sum of weighted cooccurrences by the number of cooccurrences"""

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
        decay_interval: int = 1,
        similarity: str = "cosine",
        decay_function: str = "exponential",
        event_age_weight: float = 0,
    ):

        super().__init__(K, fit_decay, predict_decay, decay_interval, similarity, decay_function)
        self.event_age_weight = event_age_weight

    def _fit(self, X: InteractionMatrix):
        num_users, num_items = X.shape

        # Get the timestamps multi index
        last_timestamps_matrix = X.last_timestamps_matrix
        now = last_timestamps_matrix.max() + 1

        # # Rescale the timestamps matrix to be in the right 'time units'
        # last_timestamps_matrix = last_timestamps_matrix
        # item_similarities = csr_matrix((num_items, num_items))
        # for user in X.active_users:
        #     # Construct user history as np array
        #     user_hist = last_timestamps_matrix[user, :].T

        #     # Compute the Cooc matrix for this user,
        #     # with the difference in timestamp as value.
        #     # 1. compute cooc matrix,
        #     #   such that cooc_one_ts[i,j] = t(j) if hist[i] and hist[j]
        #     cooc_one_ts = user_hist.astype(bool) @ (user_hist.T)

        #     # 2. construct the cooc matrix with timsteamps of item i
        #     cooc_other_ts = cooc_one_ts.astype(bool).multiply(user_hist)
        #     # By adding a small value to one of the timestamps, we avoid vanishing zero distances.
        #     cooc_other_ts.data = cooc_other_ts.data + EPSILON

        #     # 3. Construct cooc csr matrix with the time delta between interactions
        #     cooc_time_delta = csr_matrix(
        #         abs(cooc_one_ts - cooc_other_ts),
        #     )

        #     # 4. Compute the maximal timedelta with t_0
        #     cooc_distance_to_now = (cooc_one_ts < cooc_other_ts).multiply(cooc_one_ts) + (
        #         cooc_other_ts < cooc_one_ts
        #     ).multiply(cooc_other_ts)
        #     cooc_distance_to_now.data = now - cooc_distance_to_now.data

        #     # Compute similarity contribution as 1/(delta_t + delta_d)
        #     similarity_contribution = cooc_time_delta + (self.event_age_weight * cooc_distance_to_now)
        #     similarity_contribution.data = self._decay_contribution(similarity_contribution.data)

        #     item_similarities += similarity_contribution

        item_similarities = lil_matrix((X.shape[1], X.shape[1]))

        for user, hist in tqdm(X.sorted_item_history):
            for ix, context in enumerate(hist):
                context_ts = last_timestamps_matrix[user, context]
                for target in hist[ix + 1 :]:
                    target_ts = last_timestamps_matrix[user, target]
                    event_distance = abs(context_ts - target_ts)
                    max_distance_to_now = now - min(context_ts, target_ts)

                    contrib = self._decay_contribution(event_distance + self.event_age_weight * max_distance_to_now)
                    item_similarities[context, target] += contrib
                    item_similarities[target, context] += contrib

        # normalise the similarities
        if self.similarity == "hermann":
            cooc = csr_matrix(X.binary_values.T @ X.binary_values)
            item_similarities = item_similarities.multiply(invert(cooc))
        elif self.similarity == "conditional_probability":
            item_similarities = item_similarities.multiply(invert(X.binary_values.sum(axis=0)))

        # item_similarities[np.arange(num_items), np.arange(num_items)] = 0

        self.similarity_matrix_ = get_top_K_values(csr_matrix(item_similarities), self.K)

    def _decay_contribution(self, contribution):
        return self.DECAY_FUNCTIONS[self.decay_function](
            np.array([contribution / self.decay_interval]), self.fit_decay
        )
