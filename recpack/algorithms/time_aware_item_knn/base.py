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

    Where alpha is the decay scaling parameter, and age is the time between "now" and the timestamp of the event.
    "Now" is considered as the maximal timestamp in the matrix + 1s.

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
        "concave": concave_decay,
        "convex": convex_decay,
        "identity": lambda x: x,
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
            raise ValueError("Parameter decay_interval needs to be a positive integer.")

        self.decay_interval = decay_interval

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"Similarity {similarity} is not supported.")
        self.similarity = similarity

        if decay_function not in self.DECAY_FUNCTIONS:
            raise ValueError(f"Decay function {decay_function} is not supported.")

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
        """Weigh each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> InteractionMatrix:
        """Weigh each of the interactions by the decay factor of its timestamp."""
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
        """Weigh the interaction matrix based on age of the events.

        If decay is 0, it is assumed to be disabled, and so we just return binary matrix.
        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :param decay: Decay parameter, is 1/half_life.
        :type decay: float
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """

        if decay == 0:
            return X.binary_values

        timestamp_mat = X.last_timestamps_matrix
        # The maximal timestamp in the matrix is used as 'now',
        # age is encoded as now - t
        now = timestamp_mat.data.max() + 1
        timestamp_mat.data = self.DECAY_FUNCTIONS[self.decay_function](
            (now - timestamp_mat.data) / self.decay_interval, decay
        )
        return csr_matrix(timestamp_mat)


class TARSItemKNNCoocDistance(TARSItemKNN):

    # TODO: think about reasonable other similarity functions.
    SUPPORTED_SIMILARITIES = ["cooc", "conditional_probability", "hermann"]
    """Supported similarities, ``hermann`` is the similarity defined in Hermann et al. (2010),
    dividing the sum of weighted cooccurrences by the number of cooccurrences"""

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
        decay_interval: int = 1,
        similarity: str = "cooc",
        decay_function: str = "exponential",
        event_age_weight: float = 0,
    ):
        super().__init__(K, fit_decay, predict_decay, decay_interval, similarity, decay_function)
        self.event_age_weight = event_age_weight

    def _fit(self, X: InteractionMatrix):
        num_users, num_items = X.shape

        # Get the timestamps matrix, and apply the interval
        last_timestamps_matrix = X.last_timestamps_matrix / self.decay_interval
        now = last_timestamps_matrix.max() + 1 / self.decay_interval

        self.similarity_matrix_ = lil_matrix((X.shape[1], X.shape[1]))

        max_age_possible = last_timestamps_matrix.data.max() - last_timestamps_matrix.data.min()
        # Loop over all items as centers
        for i in tqdm(range(num_items)):
            n_center_occ = (last_timestamps_matrix[:, i] > 0).sum()
            if n_center_occ == 0:  # Unvisited item, no neighbours
                continue

            # Compute |t_i - t_j| for each j cooccurring with item i
            cooc_ts = last_timestamps_matrix.multiply(last_timestamps_matrix[:, i] > 0)
            distance = cooc_ts - (cooc_ts > 0).multiply(last_timestamps_matrix[:, i])
            distance.data = np.abs(distance.data)

            # Add min age of i and j to the distance computed.
            if self.event_age_weight > 0:
                broadcasted_age_of_center = (last_timestamps_matrix > 0).multiply(last_timestamps_matrix[:, i])
                target_has_smallest_age = last_timestamps_matrix < broadcasted_age_of_center
                center_has_smallest_age = (cooc_ts > 0) - target_has_smallest_age
                min_age = target_has_smallest_age.multiply(last_timestamps_matrix) + center_has_smallest_age.multiply(
                    last_timestamps_matrix[:, i]
                )
                min_age.data = now - min_age.data
                distance = distance + (distance > 0).multiply(self.event_age_weight * min_age)

            # Decay the distances
            distance.data = self.DECAY_FUNCTIONS[self.decay_function](
                distance.data, self.fit_decay, max_age=max_age_possible
            )

            similarities = csr_matrix(distance.sum(axis=0))
            # Normalisation options.
            if self.similarity == "hermann":
                n_cooc = (cooc_ts > 0).sum(axis=0)
                similarities = similarities.multiply(invert(n_cooc))
            elif self.similarity == "conditional_probability":
                similarities = similarities.multiply(1 / n_center_occ)
            self.similarity_matrix_[i] = get_top_K_values(csr_matrix(similarities), self.K)

        self.similarity_matrix_ = self.similarity_matrix_.tocsr()
