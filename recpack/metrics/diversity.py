import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix

from recpack.metrics.base import FittedMetric, ListwiseMetricK
from recpack.util import get_top_K_ranks


class IntraListDiversityK(FittedMetric, ListwiseMetricK):
    """Computes the diversity of items in a list of recommendations.

    Requires to be fit, to be used.
    Fit typically happens on a boolean metadata association matrix.
    Where each row is an item, and each column a metadata concept.
    The distance between two items is then computed as their Jaccard distance.

    For each user u, the intra list diversity is computed as

    .. math::

        \\frac{\\sum_{i \\in topK(u), j \\in topK(u) \\setdiff i} J(i,j)}{K(K-1)}

    """

    def __init__(self, K):
        super().__init__(K)

        self.X = None
        self.results_per_list = []

    def fit(self, X: csr_matrix) -> None:
        """X is an item to feature matrix, with features one hot encoded"""
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

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Compute the diversity of the predicted user preferences."""
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)
        # resolve top K items per user
        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        scores = csr_matrix(np.zeros((y_pred_top_K.shape[0], 1)))

        for u in range(0, y_pred_top_K.shape[0]):
            recommended_items = list(set(y_pred_top_K[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            scores[u, 0] = self._get_ild(recommended_items)

        self.scores_ = scores

        self.value_ = self.scores_.sum() / self.scores_.shape[0]
