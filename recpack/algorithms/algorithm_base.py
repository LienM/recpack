class Algorithm:

    @property
    def name(self):
        return None

    def fit(self, X):
        pass

    def predict(self, X):
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


class TwoMatrixFitAlgorithm(Algorithm):
    def fit(self, X_1, X_2):
        pass
