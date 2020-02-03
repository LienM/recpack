import secrets

from collections import Counter

import numpy as np
import numpy.random


class Algorithm:

    def fit(X):
        B = None
        return B

    def predict(X):
        pass

    def save(filename):
        pass

    def load(filename):
        pass


class EASE(Algorithm):

    def __init__(self, l2=1e3, B=None):
        self.B = B
        self.l2 = l2

    def fit(self, X, w=None):
        """Compute the closed form solution and then rescale using diagM(w)"""
        # Dense linear model algorithm with closed-form solution
        # Embarrassingly shallow auto-encoder from Steck @ WWW 2019
        # https://arxiv.org/pdf/1905.03375.pdf
        # Dense version in Steck et al. @ WSDM 2020
        # http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf
        # Eq. 21: B = I − P · diagMat(1 ⊘ diag(P)
        # More info on the solution for rescaling targets in section 4.2 of
        # Collaborative Filtering via High-Dimensional Regression from Steck
        # https://arxiv.org/pdf/1904.13033.pdf
        # Eq. 14 B_scaled = B * diagM(w)

        # Compute P
        P = np.linalg.inv(X.T @ X + self.l2 * np.identity((X.shape[1]), dtype=np.float32))
        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = .0

        if w is None:
            self.B = B
            return B
        else:
            B_scaled = B @ np.diag(w)
            self.B = B_scaled

            return B_scaled

    def load(self, filename):
        self.B = np.load(filename)

        return self.B

    def save(self, filename=None):
        if self.B is None:
            raise Exception("Fit a model before trying to save it, dumbass.")

        if not filename:  # TODO Check if filename is valid
            filename = './B_' + secrets.token_hex(10)

        np.save(filename, self.B)

        return filename

    def predict(self, X):
        if self.B is None:
            raise Exception("Fit a model before trying to predict with it.")
        return X @ self.B


class Random(Algorithm):

    def __init__(self):
        super().__init__()
        self.items = None

    def fit(self, X):
        self.items = list(set(X.nonzero()[1]))

    def predict(self, K):
        return numpy.random.choice(self.items, size=K, replace=False)


class Popularity(Algorithm):

    def __init__(self):
        super().__init__()
        self.sorted_items = None

    def fit(self, X):
        items = list(X.nonzero()[1])
        self.sorted_items = list(zip(*Counter(items).most_common()))[0]

    def predict(self, K):
        return self.sorted_items[0:K]


ALGORITHMS = {
    'ease': EASE,
    'random': Random
}


def get_algorithm(algorithm_name):
    return ALGORITHMS[algorithm_name]
