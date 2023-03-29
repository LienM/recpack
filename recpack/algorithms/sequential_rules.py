# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.util import invert
from recpack.matrix import Matrix, InteractionMatrix
from recpack.util import get_top_K_ranks, get_top_K_values


class SequentialRules(TopKItemSimilarityMatrixAlgorithm):
    """Recommends the item that most likely follows a user's last interaction.

    Implemented as described in
    Ludewig, M., Jannach, D.
    Evaluation of session-based recommendation algorithms.
    User Model User-Adap Inter 28, 331–390 (2018).
    https://doi.org/10.1007/s11257-018-9209-6

    Considers only cooccurrences between item i and item j, when item j was visited after item i.
    The weight of each cooccurrence is based on the number of steps to get from 1 to the next :math:`1/x`.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on. 
        Defaults to 200
    :type K: int, optional
    :param max_steps: Maximal amount of steps to look for neighbouring items. Defaults to 10.
    :type max_steps: int, optional
    """

    def __init__(self, K: int = 200, max_steps: int = 10):
        super().__init__(K)
        self.max_steps = max_steps

    def _transform_fit_input(self, X: Matrix) -> InteractionMatrix:
        """Weight each of the interactions by the decay factor of its timestamp"""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> csr_matrix:
        """Weight each of the interactions by the decay factor of its timestamp"""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)

        # Get only the last interacted item per user.
        return get_top_K_ranks(X.last_timestamps_matrix, 1)

    def _weight(self, n_steps):
        return 1 / n_steps

    def _fit(self, X: InteractionMatrix):
        num_items = X.shape[1]
        similarities = lil_matrix((num_items, num_items))

        for user, hist in X.sorted_item_history:
            hist_len = len(hist)
            for ix, context in enumerate(hist):
                for gap, target in enumerate(hist[ix + 1 : min(ix + self.max_steps + 1, hist_len)]):
                    similarities[context, target] += self._weight(n_steps=gap + 1)

        # Set self-similarity to 0
        similarities[list(range(num_items)), list(range(num_items))] = 0

        # The similarity is now finalised by normalising the sum computed above,
        # by dividing with the amount of occurrences of the left hand side item.
        # We also only keep the K highest values.
        self.similarity_matrix_ = get_top_K_values(
            csr_matrix(similarities.multiply(invert(X.binary_values.sum(axis=0).T))), self.K
        )
