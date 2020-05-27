import numpy as np
import scipy.sparse

from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.user_item_interactions_algorithms import (
    UserItemInteractionsAlgorithm,
)


def woodbury(Ainv, U, C, V):
    """ Computes (A + UCV)^-1 using precomputed A^-1"""
    Cinv = np.linalg.inv(C)
    V_Ainv_U = np.linalg.multi_dot((V, Ainv, U))
    T = np.linalg.inv(Cinv + V_Ainv_U)
    return Ainv - np.linalg.multi_dot((Ainv, U, T, V, Ainv))


class WEASE(UserItemInteractionsAlgorithm):
    def __init__(self, l2=500, alpha=0):
        super().__init__()
        self.l2 = l2
        self.alpha = alpha

    def fit(self, X, y=None):
        C = self.alpha * X

        l2_n = self.l2 * np.identity(X.shape[1])
        B = np.zeros((X.shape[1], X.shape[1]))

        Ainv = np.linalg.inv(X.T @ X + l2_n)

        for j in range(B.shape[1]):
            cj = C[:,j].toarray().squeeze() + 1
            Cj = np.diag(cj)
            xtc = X.T @ Cj

            # Choose best option to do inversion
            if X.shape[1] <= C[:,j].nnz:
                A = np.linalg.inv(xtc @ X + l2_n)
            else:
                # Option to compute A with woodbury matrix identity using:
                #  Y^TCjY = Y^TY + Y^T(Cj-I)Y, and
                #  Cj-I only contains Nu non zero elements with Nj the amount of users that like item j
                # This means the second term can be reduced to rank Nj and if Nj < N this is faster.
                d = cj - 1
                U = X[d.nonzero()].T
                V = U.T
                D = np.diag(d[d.nonzero()])
                A = woodbury(Ainv, U.toarray(), D, V.toarray())

            col = xtc @ X[:,j]

            lag = np.zeros((X.shape[1], 1))
            lag[j] = - A[j] @ col / A[j, j]

            B[:, j] = np.squeeze(A @ (col + lag))

        self.B_ = B

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores
