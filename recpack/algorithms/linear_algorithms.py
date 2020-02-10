import secrets

import numpy as np

import scipy.sparse

from sklearn.linear_model import SGDRegressor, ElasticNet

from .algorithm_base import Algorithm


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
        # Somehow Robin's local env seems to not want to make P an ndarray, and makes it a matrix
        if type(P) == np.matrix:
            P = P.A
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


class SLIM(Algorithm):
    """ Implementation of the SLIM model.
    loosely based on https://github.com/Mendeley/mrec
    """
    def __init__(self, l1_reg=0.0005, l2_reg=0.00005, fit_intercept=True, ignore_neg_weights=True, model='sgd'):

        self.similarity_matrix = None

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        # Translate regression parameters into the expected sgd parameters
        self.alpha = self.l1_reg + self.l2_reg
        self.l1_ratio = self.l1_reg / self.alpha
        self.fit_intercept = fit_intercept
        self.ignore_neg_weights = ignore_neg_weights

        # Construct internal model
        # ALLOWED MODELS:
        ALLOWED_MODELS = ['sgd']

        if model == 'sgd':
            self.model = SGDRegressor(
                penalty='elasticnet',
                fit_intercept=fit_intercept,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio
            )

        else:
            raise NotImplementedError(
                f"{model} is not yet implemented, "
                f"please use one of {ALLOWED_MODELS}"
            )

    def _compute_similarities(self, work_matrix, item):
        new_matrix = work_matrix.tocoo()
        target = new_matrix.getcol(item)
        data_indices = np.where(new_matrix.col == item)[0]
        new_matrix.data[data_indices] = 0
        self.model.fit(new_matrix, target.toarray().ravel())

        w = self.model.coef_
        if self.ignore_neg_weights:
            w[w < 0] = 0
        return w

    def fit(self, X):
        """Fit a similarity matrix based on data X.

        X is an m x n binary matrix of user item interactions.
        Where m is the number of users, and n the number of items.
        """
        # Prep sparse representation inputs
        data = []
        row = []
        col = []
        # Loop over all items
        for j in range(X.shape[1]):
            # Compute the contribution values of all other items for the item j using linear regression
            w = self._compute_similarities(X, j)
            # Update sparse repr. inputs.
            # w[i,j] = the contribution of item i to predicting item j
            for i in w.nonzero()[0]:
                data.append(w[i])
                row.append(i)
                col.append(j)

        # Construct similarity matrix.
        # Shape is determined by 2nd dimension of the shape of input matrix X
        self.similarity_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(X.shape[1], X.shape[1]))

    def predict(self, X):
        """Predict scores for each user, item pair
        X is a user item interaction matrix in sparse represenation size m' x n
        Where n needs to be the same as the n in fit, but m' can be anything.

        response will be a m' x n matrix again with predicted scores.

        No history is actually filtered.
        """
        if self.similarity_matrix is None:
            raise Exception("Fit a model before trying to predict with it.")
        # TODO, this looks exactly the same as NN's recommendation -> refactor into a similarity based class.
        scores = X @ self.similarity_matrix
        return scores.toarray()
