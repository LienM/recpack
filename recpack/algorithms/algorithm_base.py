import scipy.sparse
from sklearn.base import BaseEstimator


class Algorithm(BaseEstimator):

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = "_".join((f"{k}_{v}" for k, v in self.get_params().items()))
        return self.name + "__" + paramstring

    def __str__(self):
        return self.name

    def fit(self, X):
        pass

    def predict(self, X: scipy.sparse.csr_matrix):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class Baseline:

    def fit(self, X):
        pass

    def predict(self, X, K):
        pass