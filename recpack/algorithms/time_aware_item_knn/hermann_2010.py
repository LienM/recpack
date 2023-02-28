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
from recpack.algorithms.time_aware_item_knn.base import TARSItemKNNCoocDistance
from recpack.algorithms.time_aware_item_knn.decay_functions import InverseDecay
from recpack.algorithms.util import invert
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values


EPSILON = 1e-13


class TARSItemKNNHermann(TopKItemSimilarityMatrixAlgorithm):
    """ItemKNN algorithm with temporal decay,
    taking into account age of events and pairwise distance between cooccurences.

    Presented in Hermann, Christoph. "Time-based recommendations for lecture materials."
    EdMedia+ Innovate Learning. Association for the Advancement of Computing in Education (AACE), 2010.

    Similarity between two items is computed as the avg of :math:`S_{u,i,j}`
    for each user that has seen both items i and j.

    .. math::

        S_{u,i,j} = \\frac{1}{\\Delta t_{u,i,j} + \\Delta d_{u,i,j}}

    where :math:`\\Delta t_{u,i,j}` is the distance in time units between the user interacting with item i and j.
    :math:`\\Delta d_{u,i,j}` is the maximal distance in time units between a user interactions with i or j to now.

    :param K: number of neighbours to store while training. Defaults to 200.
    :type K: int, optional
    :param decay_interval: Size of a single time unit in seconds.
        Allows more finegrained parameters for large scale datasets where events are collected over months of data.
        Defaults to 1 (second).
    :type decay_interval: int, optional
    """

    def __init__(self, K: int = 200, decay_interval: int = 1):
        super().__init__(K=K)
        self.decay_interval = decay_interval
        self.fit_decay_func = InverseDecay()

    def _transform_fit_input(self, X: Matrix) -> InteractionMatrix:
        """Weigh each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: InteractionMatrix):
        num_users, num_items = X.shape

        # Get the timestamps matrix, and apply the interval
        last_timestamps_matrix = X.last_timestamps_matrix / self.decay_interval
        now = last_timestamps_matrix.max() + 1 / self.decay_interval

        self.similarity_matrix_ = lil_matrix((X.shape[1], X.shape[1]))

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

            broadcasted_age_of_center = (last_timestamps_matrix > 0).multiply(last_timestamps_matrix[:, i])
            target_has_smallest_age = last_timestamps_matrix < broadcasted_age_of_center
            center_has_smallest_age = (cooc_ts > 0) - target_has_smallest_age
            min_age = target_has_smallest_age.multiply(last_timestamps_matrix) + center_has_smallest_age.multiply(
                last_timestamps_matrix[:, i]
            )
            min_age.data = now - min_age.data
            distance = distance + (distance > 0).multiply(min_age)

            # Decay the distances
            distance.data = self.fit_decay_func(distance.data)

            similarities = csr_matrix(distance.sum(axis=0))
            n_cooc = (cooc_ts > 0).sum(axis=0)
            similarities = similarities.multiply(invert(n_cooc))
            self.similarity_matrix_[i] = get_top_K_values(csr_matrix(similarities), self.K)

        self.similarity_matrix_ = self.similarity_matrix_.tocsr()
