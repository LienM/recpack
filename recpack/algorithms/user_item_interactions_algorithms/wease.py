from collections import Counter

import numpy as np
import scipy.sparse

from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.user_item_interactions_algorithms import (
    UserItemInteractionsAlgorithm,
)

from tqdm.auto import tqdm


def woodbury(Ainv, U, C, V):
    """ Computes (A + UCV)^-1 using precomputed A^-1"""
    Cinv = np.linalg.inv(C)
    Ainv_U = Ainv @ U
    V_Ainv_U = np.linalg.multi_dot((V, Ainv_U))
    T = np.linalg.inv(Cinv + V_Ainv_U)
    return Ainv - np.linalg.multi_dot((Ainv_U, T, V, Ainv))


def sortKeysByValue(d: dict):
    return [x[0] for x in sorted(d.items(), key=lambda x: x[1])]


class WEASE(UserItemInteractionsAlgorithm):
    def __init__(self, l2=500, alpha=0):
        super().__init__()
        self.l2 = l2
        self.alpha = alpha

    def fit(self, X, P=None):
        """ P are preferences, X are the interactions (not necessarily binary) """
        # C confidence
        C = self.alpha * X

        l2_n = self.l2 * np.identity(P.shape[1])
        B = np.zeros((P.shape[1], P.shape[1]))

        print("Inversion of xtx")
        Ainv = np.linalg.inv(P.T @ P + l2_n)

        # iterate over columns from least dense to densest
        rows, cols = X.nonzero()
        counts = Counter(cols)
        print("items:", P.shape[1])
        print("more ratings than items", len([1 for x, y in counts.items() if y >= P.shape[1]]))
        print("less ratings than items", len([1 for x, y in counts.items() if y < P.shape[1]]))
        print("max", max(counts.values()))
        for j in tqdm(sortKeysByValue(counts)):
            cj = C[:,j].toarray().squeeze() + 1
            Cj = scipy.sparse.diags(cj, 0)
            xtc = P.T @ Cj

            print(C[:, j].nnz)

            # Choose best option to do inversion
            if P.shape[1] <= C[:, j].nnz:
                A = np.linalg.inv(xtc @ P + l2_n)
            else:
                # Option to compute A with woodbury matrix identity using:
                #  Y^TCjY = Y^TY + Y^T(Cj-I)Y, and
                #  Cj-I only contains Nu non zero elements with Nj the amount of users that like item j
                # This means the second term can be reduced to rank Nj and if Nj < N this is faster.
                d = cj - 1
                U = P[d.nonzero()].T
                V = U.T
                D = np.diag(d[d.nonzero()])
                A = woodbury(Ainv, U.toarray(), D, V.toarray())

            col = xtc @ P[:, j]

            lag = np.zeros((P.shape[1], 1))
            lag[j] = - A[j] @ col / A[j, j]

            B[:, j] = np.squeeze(A @ (col + lag))

        self.B_ = B

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores
