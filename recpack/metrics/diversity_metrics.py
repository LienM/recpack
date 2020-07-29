from recpack.metrics.metric import MetricK
from recpack.algorithms.algorithm_base import Algorithm
import numpy as np
import scipy.sparse
from scipy.spatial import distance
import logging

class IntraListDistanceK(MetricK):
    def __init__(self, K):
        self.X = None
        self.K = K
        self.distances = []

    def fit(self, X):
        """X is an item to feature matrix, 1 hot encoded"""
        self.X = X

    def _get_distance(self, i, j):
        return distance.jaccard(self.X[i].toarray()[0], self.X[j].toarray()[0])

    def _get_ild(self, recommended_items, total_items):
        # Compute the IDL for this list
        # Sum part: SUM(d(i_k, i_l) for i_k in R and l < k)
        # We will compute this sum by constructing a sparse matrix with a 1 on each of the required i_k, i_l tuples
        if len(recommended_items) <= 1:
            # If there are 1 or no items, the intra list distance is 0
            return 0
        coordinates = [(i_k, i_l) for i, i_k in enumerate(recommended_items) for i_l in recommended_items[:i]]
        distances = [self._get_distance(i, j) for i,j in coordinates]

        t_distance = sum(distances)

        ild = (2 / (len(recommended_items)*(len(recommended_items)-1))) * t_distance
        return ild

    def update(self, X_pred, X_true):
        """ Only looks at the predicted items, does not care about the 'true' items.
        """
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred.nonzero()[0]))
        
        for u in nonzero_users:
            recommended_items = list(set(X_pred_top_K[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            self.distances.append(self._get_ild(recommended_items, X_pred.shape[1]))

    @property
    def value(self):
        if len(self.distances) == 0:
            return 0
        return sum(self.distances) / len(self.distances)
