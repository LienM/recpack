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
)
from recpack.data.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values


class TARSItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """ItemKNN algorithm where older interactions have less weight during both prediction and training.

    This class is the baseclass for ItemKNN weighting approaches, combining their functionality,
    and allowing unpublished combinations of settings.
    Includes work by Liu, Nathan N., et al. (2010), Ding et al. (2005) and lee et al. (2007)

    The standard framework for all of these approaches can be summarised as:

    - When training the user interaction matrix is weighted to take into account temporal information available
    - Similarities are computed on this weighted matrix, using various similarity measures.
    - When predicting the interactions are similarly weighted, giving more weight to more recent interactions.
    - Recommendation scores are obtained by multiplying the weighted interaction matrix with
      the previously computed similarity matrix

    The default weighting in this base class is:
    .. math::

        e^{- \\alpha \\text{age}}

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
    :param similarity: Which similarity measure to use. Defaults to `"cosine"`
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
    }

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
        similarity="cosine",
        decay_function="exponential",
    ):
        # Uses other default parameters for ItemKNN
        super().__init__(K=K)
        self.fit_decay = fit_decay
        self.predict_decay = predict_decay

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} is not supported")
        self.similarity = similarity

        if decay_function not in self.DECAY_FUNCTIONS:
            raise ValueError(f"decay function {decay_function} is not supported")

        self.decay_function = decay_function

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X

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
        """Weight each of the interactions by the decay factor of its timestamp"""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> InteractionMatrix:
        """Weight each of the interactions by the decay factor of its timestamp"""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
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

        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :param decay: decay parameter, is 1/half_life
        :type decay: float
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """
        timestamp_mat = X.last_timestamps_matrix
        # The maximal timestamp in the matrix is used as 'now',
        # age is encoded as now - t
        now = timestamp_mat.data.max()
        timestamp_mat.data = self.DECAY_FUNCTIONS[self.decay_function](now - timestamp_mat.data, decay)
        return csr_matrix(timestamp_mat)
