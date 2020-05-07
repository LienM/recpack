import secrets

import numpy as np
import scipy.sparse

from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.user_item_interactions_algorithms import (
    UserItemInteractionsAlgorithm,
)


class EASE(UserItemInteractionsAlgorithm):
    def __init__(self, l2=1e3):
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
        P = np.linalg.inv(
            X.T @ X + self.l2 * np.identity((X.shape[1]), dtype=np.float32)
        )
        # Somehow Robin's local env seems to not want to make P an ndarray, and makes it a matrix
        if type(P) == np.matrix:
            P = P.A
        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = 0.0

        if w is None:
            self.B_ = scipy.sparse.csr_matrix(B)
        else:
            B_scaled = B @ np.diag(w)
            self.B_ = scipy.sparse.csr_matrix(B_scaled)

        return self

    def load(self, filename):
        self.B_ = np.load(filename)

        return self.B_

    def save(self, filename=None):
        check_is_fitted(self)

        if not filename:  # TODO Check if filename is valid
            filename = "./B_" + secrets.token_hex(10)

        np.save(filename, self.B_)

        return filename

    def predict(self, X):
        check_is_fitted(self)

        scores = X @ self.B_

        if not isinstance(scores, scipy.sparse.csr_matrix):
            scores = scipy.sparse.csr_matrix(scores)

        return scores

    @property
    def name(self):
        return f"ease_lambda_{self.l2}"


class EASE_XY(EASE):
    """ Variation of EASE where we encode Y from X (no autoencoder). """
    def fit(self, X, Y=None):
        if not Y:
            raise RuntimeError("Train regular EASE (with X=Y) using the EASE algorithm, not EASE_XY.")
        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ Y).todense()

        D = np.identity(X.shape[1]) @ np.diag(np.diag(B_rr) / np.diag(P))
        self.B_ = scipy.sparse.csr_matrix(B_rr - P @ D)

        return self


class SLIM(UserItemInteractionsAlgorithm):
    """ Implementation of the SLIM model.
    loosely based on https://github.com/Mendeley/mrec
    """

    def __init__(
        self,
        l1_reg=0.0005,
        l2_reg=0.00005,
        fit_intercept=True,
        ignore_neg_weights=True,
        model="sgd",
    ):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        # Translate regression parameters into the expected sgd parameters
        self.alpha = self.l1_reg + self.l2_reg
        self.l1_ratio = self.l1_reg / self.alpha
        self.fit_intercept = fit_intercept
        self.ignore_neg_weights = ignore_neg_weights

        # Construct internal model
        # ALLOWED MODELS:
        ALLOWED_MODELS = ["sgd"]

        if model == "sgd":
            self.model = SGDRegressor(
                penalty="elasticnet",
                fit_intercept=fit_intercept,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
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
        self.similarity_matrix_ = scipy.sparse.csr_matrix(
            (data, (row, col)), shape=(X.shape[1], X.shape[1])
        )
        return self

    def predict(self, X):
        """Predict scores for each user, item pair
        X is a user item interaction matrix in sparse represenation size m' x n
        Where n needs to be the same as the n in fit, but m' can be anything.

        response will be a m' x n matrix again with predicted scores.

        No history is actually filtered.
        """
        check_is_fitted(self)

        # TODO, this looks exactly the same as NN's recommendation -> refactor into a similarity based class.
        scores = X @ self.similarity_matrix_

        if not isinstance(scores, scipy.sparse.csr_matrix):
            scores = scipy.sparse.csr_matrix(scores)

        return scores

    @property
    def name(self):
        return f"slim_l1_{self.l1_reg}_l2_{self.l2_reg}_{self.model}"
