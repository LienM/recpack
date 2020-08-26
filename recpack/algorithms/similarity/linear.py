import secrets

import numpy as np
import scipy.sparse

from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.similarity.base import SimilarityMatrixAlgorithm


class EASE(SimilarityMatrixAlgorithm):
    def __init__(self, l2=1e3, alpha=0):
        """ l2 norm for regularization and alpha exponent to filter popularity bias. """
        super().__init__()
        self.l2 = l2
        self.alpha = alpha  # alpha exponent for filtering popularity bias

    def fit(self, X, y=None):
        """Compute the closed form solution, optionally rescalled to counter popularity bias (see param alpha). """
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
        if y is not None:
            raise RuntimeError("Train EASE_XY.")

        # Compute P
        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(
            XTX +
            self.l2 *
            np.identity(
                (X.shape[1]),
                dtype=np.float32))

        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = 0.0

        if self.alpha != 0:
            w = 1 / np.diag(XTX) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        self._check_fit_complete()
        return self

    def load(self, filename):
        self.similarity_matrix_ = np.load(filename)

        return self.B_

    def save(self, filename=None):
        check_is_fitted(self)

        if not filename:  # TODO Check if filename is valid
            filename = "./B_" + secrets.token_hex(10)

        np.save(filename, self.B_)

        return filename


class EASE_Intercept(EASE):
    """
    Variation of EASE where an item weight is learned in addition to the item-item weights.
    See footnote of Collaborative Filtering via High-Dimensional Regression from Steck
    https://arxiv.org/pdf/1904.13033.pdf
    """

    def fit(self, X, y=None):
        if y is not None:
            raise RuntimeError("Train EASE_XY.")

        y = X
        X = scipy.sparse.hstack((y, np.ones((X.shape[0], 1))))

        XTX = (X.T @ X).toarray()
        G = XTX + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(np.diag(B_rr) / np.diag(P)[:-1])
        B = B_rr
        B[:-1, :] -= P[:-1, :-1] @ D

        if self.alpha != 0:
            w = 1 / np.diag(XTX)[:-1] ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        self._check_fit_complete()

        return self

    def predict(self, X, user_ids=None):
        X = scipy.sparse.hstack((X, np.ones((X.shape[0], 1))))
        return super().predict(X, user_ids=user_ids)


class EASE_XY(EASE):
    """ Variation of EASE where we encode Y from X (no autoencoder). """

    def fit(self, X, y=None):
        if y is None:
            raise RuntimeError(
                "Train regular EASE (with X=Y) using the EASE algorithm, not EASE_XY."
            )
        XTX = X.T @ X
        G = XTX + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(np.diag(B_rr) / np.diag(P))
        B = B_rr - P @ D

        if self.alpha != 0:
            w = 1 / np.diag(XTX) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        self._check_fit_complete()

        return self


def normalize(X):
    return scipy.sparse.csr_matrix(
        scipy.sparse.diags(1 / np.sum(X, axis=1).A1) @ X)


class EASE_AVG(EASE):
    """ Variation of EASE where we take the average of weights rather than the sum (unpublished). """

    def __init__(self, l2=0.2):
        super().__init__(l2, alpha=0)

    def fit(self, X, y=None):
        if y is not None:
            raise RuntimeError("Train EASE_XY for distinct y.")
        y = X
        X = normalize(y)

        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag((1 - np.diag(B_rr)) / -np.diag(P))
        B = B_rr - P @ D
        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        self._check_fit_complete()

        return self

    def predict(self, X, user_ids=None):
        X = normalize(X)
        return super().predict(X, user_ids=user_ids)


class EASE_AVG_Int(EASE_AVG):
    """ Variation of EASE where we take the average of weights rather than the sum (unpublished)
    with unary weights for items. """

    def fit(self, X, y=None):
        if y is not None:
            raise RuntimeError("Train EASE_XY for distinct y.")
        y = X

        X = scipy.sparse.hstack((X, np.ones((X.shape[0], 1))))
        X = normalize(X)

        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(1 - np.diag(B_rr) / -np.diag(P)[:-1])
        B = B_rr
        B[:-1, :] -= P[:-1, :-1] @ D

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)
        self._check_fit_complete()
        return self

    def predict(self, X, user_ids=None):
        X = scipy.sparse.hstack((X, np.ones((X.shape[0], 1))))
        return super().predict(X, user_ids=user_ids)


class SLIM(SimilarityMatrixAlgorithm):
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
        super().__init__()

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
            # Compute the contribution values of all other items for the item j
            # using linear regression
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

        self._check_fit_complete()

        return self
