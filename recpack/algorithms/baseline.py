from collections import Counter
import random
import sys

import numpy as np
from scipy.sparse import csr_matrix


from recpack.algorithms.base import Algorithm


class Random(Algorithm):
    """Uniform random algorithm, each item has an equal chance of getting recommended.

    Simple baseline, recommendations are sampled uniformly without replacement
    from the items that were interacted with in the matrix provided to fit.
    Scores are given based on sampling rank, such that the items first
    in the sample has the highest score

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import Random

        X = csr_matrix(np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0]]))

        # There are only 3 items to recommend,
        # so can't recommend the default K=200
        algo = Random(K=3)
        # Fit algorithm, stores the nonzero items in matrix X
        # as potential items to sample during predict
        algo.fit(X)

        # Get random recos for each nonzero user
        predictions = algo.predict(X)
        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()


    :param K: How many items to sample for recommendation, defaults to 200
    :type K: int, optional
    :param seed: Seed for the random number generator used, defaults to None
    :type seed: int, optional
    :param use_only_interacted_items: Should only items visited in the training
        matrix be used to recommend from. If False all items will be recommended
        uniformly at random.
        Defaults to True.
    :type use_only_interacted_items: boolean, optional
    """

    def __init__(self, K=200, seed=None, use_only_interacted_items=True):
        super().__init__()
        self.items = None
        self.K = K
        self.use_only_interacted_items = use_only_interacted_items

        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        self.seed = seed

    def _fit(self, X: csr_matrix):
        if self.use_only_interacted_items:
            self.items_ = list(set(X.nonzero()[1]))
        else:
            self.items_ = list(np.arange(X.shape[1]))

    def _predict(self, X: csr_matrix):
        """Predict K random scores for items per row in X

        Returns numpy array of the same shape as X,
        with non zero scores for K items per row.
        """

        # For each user choose random K items, and generate a score for these items
        # Then create a matrix with the scores on the right indices
        U = X.nonzero()[0]

        score_list = [
            (u, i, random.random())
            for u in set(U)
            for i in np.random.choice(self.items_, size=self.K, replace=False)
        ]
        user_idxs, item_idxs, scores = list(zip(*score_list))
        score_matrix = csr_matrix((scores, (user_idxs, item_idxs)), shape=X.shape)

        return score_matrix


class Popularity(Algorithm):
    """Baseline algorithm recommending the most popular items in training data.

    During training the occurrences of each item is counted,
    and then normalized by dividing each count by the max count over items.


    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import Popularity

        X = csr_matrix(np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0]]))

        # There are only 3 items to recommend,
        # so can't recommend the default K=200
        algo = Popularity(K=3)
        # Fit algorithm, computes the popularities per item in X
        algo.fit(X)

        # Get the most popular items per user
        predictions = algo.predict(X)

        # Popular items are the same for all users
        assert predictions[0, 0] == predictions[1, 0]

        # All scores are in the range 0 to 1
        assert (predictions.toarray() >= 0).all()
        assert (predictions.toarray() <= 1).all()

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()


    :param K: How many items to recommend when predicting, defaults to 200
    :type K: int, optional
    """

    def __init__(self, K: int = 200):
        super().__init__()
        self.K = K

    def _fit(self, X: csr_matrix):
        #  Values in the matrix X are considered as counts of visits
        #  If your data contains ratings, you should make them binary before fitting
        items = list(X.nonzero()[1])
        sorted_scores = Counter(items).most_common()
        self.sorted_scores_ = [
            (item, score / sorted_scores[0][1]) for item, score in sorted_scores
        ]

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """For each user predict the K most popular items"""
        items, values = zip(*self.sorted_scores_[: self.K])

        users = set(X.nonzero()[0])

        U, I, V = [], [], []

        for user in users:
            U.extend([user] * self.K)
            I.extend(items)
            V.extend(values)

        score_matrix = csr_matrix((V, (U, I)), shape=X.shape)
        return score_matrix
