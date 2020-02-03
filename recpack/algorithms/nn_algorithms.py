
from .algorithm_base import Algorithm


class ItemKNN(Algorithm):

    def __init__(self, K):
        self.K = K

    def fit(self, X):
        self.items = list(set(X.nonzero()[1]))

    def predict(self, K):
        return np.random.choice(self.items, size=K, replace=False)