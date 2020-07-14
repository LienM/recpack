from collections import Counter, defaultdict

import numpy as np
import scipy.sparse
import numpy.random
import random
from .algorithm_base import Algorithm

from sklearn.utils.validation import check_is_fitted


class Random(Algorithm):
    def __init__(self, K=200, seed=None):
        super().__init__()
        self.items = None
        self.K = K
        self.seed = seed

        # TODO: Do we need this in the predict method, or is this enough.
        if self.seed is not None:
            random.seed(self.seed)

    def fit(self, X):
        self.items_ = list(set(X.nonzero()[1]))
        return self

    def predict(self, X: scipy.sparse.csr_matrix, user_ids=None):
        """Predict K random scores for items per row in X

        Returns numpy array of the same shape as X, with non zero scores for K items per row.
        """
        check_is_fitted(self)

        # For each user choose random K items, and generate a score for these items
        # Then create a matrix with the scores on the right indices
        U = X.nonzero()[0]

        score_list = [
            (u, i, random.random())
            for u in set(U)
            for i in np.random.choice(self.items_, size=self.K, replace=False)
        ]
        user_idxs, item_idxs, scores = list(zip(*score_list))
        score_matrix = scipy.sparse.csr_matrix(
            (scores, (user_idxs, item_idxs)), shape=X.shape
        )
        return score_matrix


class Popularity(Algorithm):
    def __init__(self, K=200):
        super().__init__()
        self.K = K

    def fit(self, X, y=None):
        items = list(X.nonzero()[1])
        sorted_scores = Counter(items).most_common()
        self.sorted_scores_ = [(item, score / sorted_scores[0][1]) for item, score in sorted_scores]
        return self

    def predict(self, X, user_ids=None):
        """For each user predict the K most popular items"""
        check_is_fitted(self)

        items, values = zip(*self.sorted_scores_[: self.K])

        users = set(X.nonzero()[0])

        U, I, V = [], [], []

        for user in users:
            U.extend([user] * self.K)
            I.extend(items)
            V.extend(values)

        score_matrix = scipy.sparse.csr_matrix((V, (U, I)), shape=X.shape)
        return score_matrix

    def multiply(self, value: float):
        self.sorted_scores_ = [(item, score*value) for item, score in self.sorted_scores_]

    def add(self, other):
        addition_map = defaultdict(float)
        for item, score in self.sorted_scores_:
            addition_map[item] += score
        
        for item, score in other.sorted_scores_:
            addition_map[item] += score
        
        self.sorted_scores_ = sorted(addition_map.items(), key=lambda x: x[1], reverse=True)
