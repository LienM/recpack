import secrets

import numpy as np


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

    def fit(self, X):
        # Dense linear model algorithm with closed-form solution
        # Embarrassingly shallow auto-encoder from Steck @ WWW 2019
        # https://arxiv.org/pdf/1905.03375.pdf
        # Dense version in Steck et al. @ WSDM 2020
        # http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf
        # Eq. 21: B = I − P · diagMat(1 ⊘ diag(P)
        # Compute P
        P = np.linalg.inv(X.T @ X + self.l2 * np.identity((X.shape[1]), dtype=np.float32))
        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = .0

        self.B = B

        return B

    def load(self, filename):
        self.B = np.load(filename)

        return self.B

    def save(self, filename=None):
        if not self.B:
            raise Exception("Fit a model before trying to save it, dumbass.")

        if not filename:  # TODO Check if filename is valid
            filename = './B_' + secrets.token_hex(10)

        np.save(filename, self.B)

        return filename

    def predict(self, X):
        if not self.B:
            raise Exception("Fit a model before trying to predict with it.")
        return X @ self.B


ALGORITHMS = {
    'ease': EASE
}


def get_algorithm(algorithm_name):
    return ALGORITHMS[algorithm_name]
