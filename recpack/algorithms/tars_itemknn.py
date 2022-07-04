"""Module with time-dependent ItemKNN implementations"""

import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.nearest_neighbour import (
    compute_conditional_probability,
    compute_cosine_similarity,
)
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values


class TARSItemKNNLiu(TopKItemSimilarityMatrixAlgorithm):
    """ItemKNN algorithm where older interactions have
    less weight during both prediction and training.

    Algorithm as defined in Liu, Nathan N., et al.
    "Online evolutionary collaborative filtering."
    Proceedings of the fourth ACM conference on Recommender systems. 2010.

    Each interaction is weighed as

    .. math::

        e^{- \\alpha \\text{age}}

    Where alpha is the decay scaling parameter,
    and age is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    Similarity is computed on this weighted matrix, using cosine similarity.

    At prediction time a user's history is weighted using the same formula with a different alpha.
    This weighted history is then multiplied with the precomputed similarity matrix.

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

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
    ):
        super().__init__(K)
        self.fit_decay = fit_decay
        self.predict_decay = predict_decay

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
        X = self._add_decay_to_interaction_matrix(X, self.fit_decay)

        item_similarities = compute_cosine_similarity(X)
        item_similarities = get_top_K_values(item_similarities, self.K)

        self.similarity_matrix_ = item_similarities

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
        timestamp_mat.data = np.exp(-decay * (now - timestamp_mat.data))
        return csr_matrix(timestamp_mat)


class TARSItemKNN(TARSItemKNNLiu):
    """ItemKNN algorithm where older interactions have
    less weight during both prediction and training.

    Contains extensions on the algorithm presented in Liu, Nathan N., et al.
    "Online evolutionary collaborative filtering."
    Proceedings of the fourth ACM conference on Recommender systems. 2010.

    Each interaction is weighed as

    .. math::

        e^{- \\alpha \\text{age}}

    Where alpha is the decay scaling parameter,
    and age is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    Similarity is computed on this weighted matrix, using either of the supported similarity measures.

    At prediction time a user's history is weighted using the same formula with a different alpha.
    This weighted history is then multiplied with the precomputed similarity matrix.

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

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
        similarity="cosine",
    ):
        # Uses other default parameters for ItemKNN
        super().__init__(K=K, fit_decay=fit_decay, predict_decay=predict_decay)

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
        X = self._add_decay_to_interaction_matrix(X, self.fit_decay)

        if self.similarity == "cosine":
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X)

        item_similarities = get_top_K_values(item_similarities, self.K)

        self.similarity_matrix_ = item_similarities
