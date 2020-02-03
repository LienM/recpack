from collections import Counter

import numpy as np
import numpy.random

from .algorithm_base import Baseline


class Random(Baseline):

    def __init__(self):
        self.items = None

    def fit(self, X):
        self.items = list(set(X.nonzero()[1]))

    def predict(self, X, K):
        return np.random.choice(self.items, size=K, replace=False)


class Popularity(Baseline):

    def __init__(self):
        self.sorted_items = None

    def fit(self, X):
        items = list(X.nonzero()[1])
        self.sorted_items = list(zip(*Counter(items).most_common()))[0]

    def predict(self, X, K):
        # TODO Return as many as X
        return self.sorted_items[0:K]
