# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm.auto import tqdm

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.nearest_neighbour import (
    compute_conditional_probability,
    compute_cosine_similarity,
    compute_pearson_similarity,
)
from recpack.algorithms.time_aware_item_knn.decay_functions import (
    ExponentialDecay,
    LogDecay,
    LinearDecay,
    ConcaveDecay,
    ConvexDecay,
    InverseDecay,
    NoDecay,
)
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values

EPSILON = 1e-13


class TARSItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Framework for time aware variants of the ItemKNN algorithm.

    This class was inspired by works from Liu, Nathan N., et al. (2010), Ding et al. (2005) and Lee et al. (2007).

    The framework for these approaches can be summarised as:

    - When training the user interaction matrix is weighted to take into account temporal information.
    - Similarities are computed on this weighted matrix, using various similarity measures.
    - When predicting the interactions are similarly weighted, giving more weight to more recent interactions.
    - Recommendation scores are obtained by multiplying the weighted interaction matrix with
      the previously computed similarity matrix.

    The similarity between items is based on their decayed interaction vectors:

    .. math::

        \\text{sim}(i,j) = s(\\Gamma(A_i), \\Gamma(A_j))

    Where :math:`s` is a similarity function (like ``cosine``),
    :math:`\\Gamma` a decay function (like ``exponential_decay``) and
    :math:`A_i` contains the distances to now from when the users interacted with item `i`,
    if they interacted with the item at all (else the value is 0).

    During computation, 'now' is considered as the maximal timestamp in the matrix + 1.
    As such the age is always a positive non-zero value.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, Optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to `` 1 / (24 * 3600)`` (one day).
    :type fit_decay: float, optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to ``1 / (24 * 3600)`` (one day).
    :type predict_decay: float, optional
    :param decay_interval: Size of a single time unit in seconds.
        Allows more finegrained parameters for large scale datasets where events are collected over months of data.
        Defaults to 1 (second).
    :type decay_interval: int, optional
    :param similarity: Which similarity measure to use. Defaults to ``"cosine"``.
        ``["cosine", "conditional_probability", "pearson"]`` are supported.
    :type similarity: str, Optional
    :param decay_function: The decay function to use, defaults to ``"exponential"``.
        Supported values are ``["exponential", "log", "linear", "concave", "convex", "inverse"]``
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "pearson"]
    DECAY_FUNCTIONS = {
        "exponential": ExponentialDecay,
        "log": LogDecay,
        "linear": LinearDecay,
        "concave": ConcaveDecay,
        "convex": ConvexDecay,
        "inverse": InverseDecay,
    }

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / (24 * 3600),
        predict_decay: float = 1 / (24 * 3600),
        decay_interval: int = 1,
        similarity: str = "cosine",
        decay_function: str = "exponential",
    ):
        # Uses other default parameters for ItemKNN
        super().__init__(K=K)

        if decay_interval <= 0 or type(decay_interval) == float:
            raise ValueError("Parameter decay_interval needs to be a positive integer.")

        self.decay_interval = decay_interval

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"Similarity {similarity} is not supported.")
        self.similarity = similarity

        if decay_function not in self.DECAY_FUNCTIONS:
            raise ValueError(f"Decay function {decay_function} is not supported.")

        self.decay_function = decay_function

        # Verify decay parameters
        if self.decay_function in ["exponential", "log", "linear", "concave", "convex"]:
            if fit_decay != 0:
                self.DECAY_FUNCTIONS[decay_function].validate_decay(fit_decay)

            if predict_decay != 0:
                self.DECAY_FUNCTIONS[decay_function].validate_decay(predict_decay)

        self.fit_decay = fit_decay
        self.predict_decay = predict_decay
        self.decay_function = decay_function

    def _get_decay_func(self, decay, max_value):
        if decay == 0:
            return NoDecay()

        elif self.decay_function == "inverse":
            return self.DECAY_FUNCTIONS[self.decay_function]()
        elif self.decay_function in ["exponential", "convex"]:
            return self.DECAY_FUNCTIONS[self.decay_function](decay)
        elif self.decay_function in ["log", "linear", "concave"]:
            return self.DECAY_FUNCTIONS[self.decay_function](decay, max_value)

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X.

        Scores are computed by matrix multiplication of weighted X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        X = self._add_decay_to_predict_matrix(X)
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
        X = self._add_decay_to_fit_matrix(X)

        if self.similarity == "cosine":
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X)
        elif self.similarity == "pearson":
            item_similarities = compute_pearson_similarity(X)

        item_similarities = get_top_K_values(item_similarities, K=self.K)

        self.similarity_matrix_ = item_similarities

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix, decay: float) -> csr_matrix:
        """Weigh the interaction matrix based on age of the events.

        If decay is 0, it is assumed to be disabled, and so we just return binary matrix.
        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """
        timestamp_mat = X.last_timestamps_matrix
        # To get 'now', we add 1 to the maximal timestamp. This makes sure there are no vanishing zeroes.
        now = timestamp_mat.data.max() + 1
        ages = (now - timestamp_mat.data) / self.decay_interval
        timestamp_mat.data = self._get_decay_func(decay, ages.max())(ages)
        return csr_matrix(timestamp_mat)

    def _add_decay_to_fit_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X, self.fit_decay)

    def _add_decay_to_predict_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X, self.predict_decay)


class TARSItemKNNCoocDistance(TARSItemKNN):
    """Framework for time aware variants of ItemKNN that consider the time between two interactions
    when computing similarity between two items.

    Cooc similarity between two items is computed as

    .. math ::

        \\text{sim}(i,j) = \\sum\\limits_{u \\in U}[R_{ui} \\cdot R_{uj} \\cdot \\Gamma(|T_{ui} - T_{uj}|)]

    Conditional Probability based similarity is computed as

    .. math ::

        \\text{sim}(i,j) = \\frac{1}{\\sum\\limits_{u \\in U}R_{ui}} \\sum\\limits_{u \\in U}[R_{ui} \\cdot R_{uj} \\cdot \\Gamma(|T_{ui} - T_{uj}|)]

    Where :math:`\\Gamma()` is a decay function, :math:`T_{ui}` is the timestamp at which user :math:`u`
    last visited item :math:`i` and :math:`R_{ui}` indicates whether user :math:`u` interacted with item :math:`i`.
    Timestamps are in multiples of ``decay_interval``, by default in seconds.


    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to 1 / (24 * 3600).
    :type fit_decay: float, optional
    :param decay_interval: Size of a single time unit in seconds.
        Allows more finegrained parameters for large scale datasets where events are collected over months of data.
        Defaults to 1 (second).
    :type decay_interval: int, optional
    :param similarity: Which similarity measure to use, ``["cooc", "conditional_probability"]`` are supported.
        Defaults to "cooc".
    :type similarity: str, optional
    :param decay_function: Decay function to use.
        Supported values are ``["exponential", "log", "linear", "concave", "convex", "inverse"]``.
        Defaults to "exponential"
    :type decay_function: str, optional
    """

    SUPPORTED_SIMILARITIES = ["cooc", "conditional_probability"]
    """Supported similarities are ``"cooc"`` and ``"conditional_probability"``."""

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / (24 * 3600),
        decay_interval: int = 1,
        similarity: str = "cooc",
        decay_function: str = "exponential",
    ):
        super().__init__(K, fit_decay, 0, decay_interval, similarity, decay_function)

    def _fit(self, X: InteractionMatrix):
        num_users, num_items = X.shape

        # Get the timestamps matrix, and apply the interval
        last_timestamps_matrix = X.last_timestamps_matrix / self.decay_interval

        self.similarity_matrix_ = lil_matrix((X.shape[1], X.shape[1]))

        max_distance_possible = last_timestamps_matrix.data.max() - last_timestamps_matrix.data.min()
        decay_func = self._get_decay_func(self.fit_decay, max_distance_possible)

        # Loop over all items as centers
        for i in tqdm(range(num_items)):
            n_center_occ = (last_timestamps_matrix[:, i] > 0).sum()
            if n_center_occ == 0:  # Unvisited item, no neighbours
                continue

            # Compute |t_i - t_j| for each j cooccurring with item i
            cooc_ts = last_timestamps_matrix.multiply(last_timestamps_matrix[:, i] > 0)
            distance = cooc_ts - (cooc_ts > 0).multiply(last_timestamps_matrix[:, i])
            distance.data = np.abs(distance.data)

            # Decay the distances
            # We use the max_distance_possible, because the decay function needs an overall value
            # and we only give it values per user.
            distance.data = decay_func(distance.data)

            similarities = csr_matrix(distance.sum(axis=0))
            # Normalisation options.
            if self.similarity == "conditional_probability":
                similarities = similarities.multiply(1 / n_center_occ)
            else:
                # Just use the sum of the similarities (as in Xia 2010)
                pass
            self.similarity_matrix_[i] = get_top_K_values(csr_matrix(similarities), self.K)

        self.similarity_matrix_ = self.similarity_matrix_.tocsr()
