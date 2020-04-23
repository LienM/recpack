import scipy.sparse


class Algorithm:

    @property
    def name(self):
        return None

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