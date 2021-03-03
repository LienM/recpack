from collections import Counter
import random

import numpy as np
import scipy.sparse
import numpy.random


from recpack.algorithms.base import Algorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class Random(Algorithm):
    def __init__(self, K=200, seed=None):
        super().__init__()
        self.items = None
        self.K = K
        self.seed = seed

        # TODO: mention this choice in docstring
        #  -> predicting twice will not give same results.
        #  -> predicting on two new instances with same seed will give same results.
        if self.seed is not None:
            random.seed(self.seed)

    def _fit(self, X: Matrix):
        X = to_csr_matrix(X)
        self.items_ = list(set(X.nonzero()[1]))

    def _predict(self, X: Matrix):
        """Predict K random scores for items per row in X

        Returns numpy array of the same shape as X,
        with non zero scores for K items per row.
        """
        X = to_csr_matrix(X)

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

    def _fit(self, X: Matrix):
        #  Values in the matrix X are considered as counts of visits
        #  If your data contains ratings, you should make them binary before fitting
        X = to_csr_matrix(X)
        items = list(X.nonzero()[1])
        sorted_scores = Counter(items).most_common()
        self.sorted_scores_ = [
            (item, score / sorted_scores[0][1]) for item, score in sorted_scores
        ]

    def _predict(self, X: Matrix):
        """For each user predict the K most popular items"""
        X = to_csr_matrix(X)

        items, values = zip(*self.sorted_scores_[: self.K])

        users = set(X.nonzero()[0])

        U, I, V = [], [], []

        for user in users:
            U.extend([user] * self.K)
            I.extend(items)
            V.extend(values)

        score_matrix = scipy.sparse.csr_matrix((V, (U, I)), shape=X.shape)
        return score_matrix
