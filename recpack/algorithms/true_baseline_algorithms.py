from collections import Counter

import numpy as np
import scipy as sp
import scipy.sparse
import numpy.random
import random
from .algorithm_base import Algorithm


class Random(Algorithm):

    def __init__(self, K=200, seed=None):
        self.items = None
        self.K = K
        self.seed = seed

        # TODO: Do we need this in the predict method, or is this enough.
        if self.seed is not None:
            random.seed(self.seed)

    def fit(self, X):
        self.items = list(set(X.nonzero()[1]))

    def predict(self, X):
        """Predict K random scores for items per row in X

        Returns numpy array of the same shape as X, with non zero scores for K items per row.
        """
        # For each user choose random K items, and generate a score for these items
        # Then create a matrix with the scores on the right indices
        score_list = [
            (u, i, random.random())
            for u in range(X.shape[0])
            for i in np.random.choice(self.items, size=self.K, replace=False)
        ]
        user_idxs, item_idxs, scores = list(zip(*score_list))
        score_matrix = sp.sparse.csr_matrix((scores, (user_idxs, item_idxs)), shape=X.shape)
        return score_matrix.toarray()

    @property
    def name(self):
        return f"random_{self.K}"


class Popularity(Algorithm):

    def __init__(self, K=200):
        self.sorted_scores = None
        self.K = K

    def fit(self, X):
        items = list(X.nonzero()[1])
        self.sorted_scores = Counter(items).most_common()

    def predict(self, X):
        """For each user predict the K most popular items"""
        score_list = np.zeros((X.shape[1]))
        for i, s in self.sorted_scores[:self.K]:
            score_list[i] = s
        return np.repeat([score_list], X.shape[0], axis=0)

    @property
    def name(self):
        return f"popularity_{self.K}"
