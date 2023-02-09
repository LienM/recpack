# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix

from recpack.metrics.base import FittedMetric, ListwiseMetricK
from recpack.util import get_top_K_ranks


class IntraListDiversityK(FittedMetric, ListwiseMetricK):
    """Computes the diversity of items in a list of Top-K recommendations.

    To compute the intra-list diversity, the metric first needs
    to be fitted on a boolean item-feature matrix.
    This matrix should have a row for each item and a column for each feature.

    For each user u, the intra-list diversity is computed as

    .. math::

        \\frac{\\sum\\limits_{i,j \\in Top-K(u) \\\\ i \\neq j} J(i,j)}{K(K-1)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

        self.X = None
        self.results_per_list = []

    def fit(self, X: csr_matrix) -> None:
        """
        Fit a item-feature matrix that is used to determine the diversity of the list
        of Top-K recommendations. 

        :param X: Item-feature matrix.
        :type X: csr_matrix
        """
        self.X = X

    def _get_distance(self, i, j):
        return distance.jaccard(self.X[i].toarray()[0], self.X[j].toarray()[0])

    def _get_ild(self, recommended_items):
        # Compute the ILD for this list
        #  Sum part: SUM(d(i_k, i_l) for i_k in R and l < k)
        if len(recommended_items) <= 1:
            # If there are 1 or no items, the intra list distance is 0
            return 0
        coordinates = [
            (i_k, i_l)
            for i, i_k in enumerate(recommended_items)
            for i_l in recommended_items[:i]
        ]
        distances = [self._get_distance(i, j) for i, j in coordinates]

        t_distance = sum(distances)

        ild = (2 / (len(recommended_items) * (len(recommended_items) - 1))) * t_distance
        return ild

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        """Compute the diversity of the predicted user preferences."""

        scores = csr_matrix(np.zeros((y_pred_top_K.shape[0], 1)))

        for u in range(0, y_pred_top_K.shape[0]):
            recommended_items = list(set(y_pred_top_K[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            scores[u, 0] = self._get_ild(recommended_items)

        self.scores_ = scores
