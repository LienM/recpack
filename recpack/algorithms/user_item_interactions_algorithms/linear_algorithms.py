import secrets

import numpy as np
import scipy.sparse

from tqdm.auto import tqdm

from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted

from recpack.utils.monitor import Monitor
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

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores


class EASE_XY(EASE):
    """ Variation of EASE where we encode Y from X (no autoencoder). """
    def fit(self, X, y=None):
        if y is None:
            raise RuntimeError("Train regular EASE (with X=Y) using the EASE algorithm, not EASE_XY.")
        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(np.diag(B_rr) / np.diag(P))
        self.B_ = scipy.sparse.csr_matrix(B_rr - P @ D)

        return self


def normalize(X):
    return scipy.sparse.csr_matrix(scipy.sparse.diags(1/np.sum(X, axis=1).A1) @ X)


class EASE_AVG(EASE):
    """ Variation of EASE where we encode Y from X (no autoencoder). """
    def fit(self, X, y=None):
        if y is not None:
            raise RuntimeError("Train EASE_XY for distinct y.")
        y = X
        X = normalize(y)
        
        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag((1 - np.diag(B_rr)) / -np.diag(P))
        self.B_ = scipy.sparse.csr_matrix(B_rr - P @ D)

        return self

    def predict(self, X, user_ids=None):
        X = normalize(X)
        return super().predict(X, user_ids=user_ids)


class EASE_VP(UserItemInteractionsAlgorithm):
    """ Variation of EASE for views and purchases. """
    def __init__(self, l2v=300, l2p=100):
        self.l2v = l2v
        self.l2p = l2p

    def fit(self, X, y=None):
        # X are views, y are purchases
        if y is None:
            raise RuntimeError("Train regular EASE (with X=Y) using the EASE algorithm.")

        Bv = EASE(l2=self.l2v).fit(X).B_

        VminP = X - y
        Bp = EASE_XY(l2=self.l2p).fit(y, VminP).B_

        self.B_ = scipy.sparse.csr_matrix(np.vstack((Bv, -Bp)))

        return self

    def predict(self, X: scipy.sparse.csr_matrix, user_ids=None):
        # X should be hstack of X and y
        check_is_fitted(self)

        scores = X @ self.B_

        return scores


# TODO: rename to multimodal EASE? Uses different interaction types to train model
class DurabEASE(UserItemInteractionsAlgorithm):
    def __init__(self, l2v=300, l2p=100):
        self.l2v = l2v
        self.l2p = l2p

    def fit(self, X, y=None):
        if y is None:
            raise ValueError(f"Y required for {self.__class__.__name__}")
        if X.shape[1] != y.shape[1]:
            raise ValueError("X and y should have the same amount of items.")

        # Idea 1
        # learn B1: V -> P (or V -> V?)
        # learn B2: P -> V - P (does not need zero diagonal)
        # B1 encodes similarity
        # B2 encodes alternatives that are not bought together
        # predict: V @ B1 - alpha * P @ B2

        # Idea 2
        # could also optimize B1 first and then: B2 = argmin || P - (V @ B1 - P @ B2)  || + lambda_2 * || B2 ||

        # Idea 3
        # Idea 2, but use ALS method to optimize further (train B1 then B2 then B1 again, ...)
        # This is a more efficient approximation of Idea 4
        # (matrix inv of IxI instead of 2Ix2I is much less expensive, which compensates for iterative optimization)

        # Idea 4
        # stack V and P horizontally and B1 and B2 vertically. This can be optimized in closed form.
        monitor = Monitor("DurabEASE")

        monitor.update("Merge interactions")
        X_ext = scipy.sparse.hstack((X, y), format="csr")

        monitor.update("Calculate G")
        # creates diagonal matrix with first half of diagonal elements l2v and second half l2p
        reg = np.diag([self.l2v] * X.shape[1] + [self.l2p] * y.shape[1])
        G = X_ext.T @ X_ext + reg

        monitor.update("Invert G")
        P = np.linalg.inv(G)
        del G       # free memory

        monitor.update("Calculate  Brr")
        B_rr = P @ (X_ext.T @ y).todense()

        # calculate lagrangian multipliers
        monitor.update("Calculate Lagr. Mult. (prepr.)")

        # first do one pass over itemsets to find indices
        item_indices = {j: [j, j + X.shape[1]] for j in range(B_rr.shape[1])}

        monitor.update("Calculate Lagr. Mult. (invert)")
        # then calculate multipliers
        LM = np.zeros(B_rr.shape)
        for j in tqdm(range(LM.shape[1])):
            S = item_indices[j]

            P_ss = P[np.ix_(S, S)]
            B_rr_sj = B_rr[S, j]

            # pinv to account for numerical errors
            LM_sj = np.linalg.pinv(P_ss) @ B_rr_sj

            LM[S, j] = np.squeeze(LM_sj)

        monitor.update("Calculate  B")
        B = B_rr - P @ LM

        self.B_ = scipy.sparse.csr_matrix(B)
        return self

    def predict(self, X: scipy.sparse.csr_matrix, user_ids=None):
        # X should be hstack of X and y
        check_is_fitted(self)

        scores = X @ self.B_

        return scores


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

    def predict(self, X, user_ids=None):
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
