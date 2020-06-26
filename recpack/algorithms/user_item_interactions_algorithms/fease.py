import numpy as np
import scipy.sparse

from tqdm.auto import tqdm

from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted

from recpack.utils.monitor import Monitor
from recpack.algorithms.user_item_interactions_algorithms import (
    UserItemInteractionsAlgorithm,
)


class FEASE(UserItemInteractionsAlgorithm):
    def __init__(self, k=10, iterations=10, l2=1):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.iterations = iterations

    def fit(self, X, w=None):

        print("Calculate XTX")
        l2_k = self.l2 * np.identity(self.k)
        xtx = (X.T @ X).todense()
        print("Invert XTX")
        xtx_inv = np.linalg.pinv(xtx)

        print("Init Q")
        Q = np.random.random((self.k, X.shape[1]))

        print("Start iterations")
        for i in range(self.iterations):
            print("Iteration", i)
            P = xtx_inv @ xtx @ Q.T @ np.linalg.inv(Q @ Q.T + l2_k)
            Q = np.linalg.inv(P.T @ xtx @ P + l2_k) @ P.T @ xtx
            B = P @ Q

            if hasattr(self, "B_"):
                change = np.sum(np.abs(self.B_ - B))
                print("change", change)

            self.B_ = B
            self.P_ = P
            self.Q_ = Q

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores


class FEASEAdd(UserItemInteractionsAlgorithm):
    def __init__(self, k=2, iterations=10, l2=1):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.iterations = iterations

    def fit(self, X, w=None):

        print("Calculate XTX")
        l2_n = self.l2 * np.identity(X.shape[1])
        l2_k = self.l2 * np.identity(self.k)
        xtx = (X.T @ X).todense()
        xtx_minl2 = xtx - l2_n

        print("Init Q")
        Q = np.random.random((self.k, X.shape[1]))

        print("Start iterations")
        i_knl2 = self.l2 * np.identity((X.shape[1] * self.k))
        for i in range(self.iterations):
            print("Iteration", i)
            P = vec_trick(xtx, Q@Q.T, xtx_minl2 @ Q.T, self.l2)
            # print("zero", xtx @ P @ Q @ Q.T + self.l2 * P - xtx_minl2 @ Q.T)

            Q = np.linalg.inv(P.T @ xtx @ P + self.l2 * l2_k) @ P.T @ xtx_minl2
            B = P @ Q

            if hasattr(self, "B_"):
                change = np.sum(np.abs(self.B_ - B))
                print("change", change)

            self.B_ = B
            self.P_ = P
            self.Q_ = Q

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores


def diag_XXT_inv(X):
    return scipy.sparse.diags(1/np.sum(X, axis=1).A1)


def vec_trick(A, B, C, l2):
    """ Solve AXB + lambda X = C """
    i_knl2 = l2 * np.identity((B.shape[0] * A.shape[0]))
    M = np.kron(B, A) + i_knl2
    c = C.flatten('F').T
    p = np.linalg.solve(M, c)
    # print("zero", A @ p - c)
    P = p.T.reshape((A.shape[0], B.shape[0]), order='F')
    return P


def solve_AXB_c(A, B, c):
    """ solve diag(AXB) = c """
    T = np.empty((A.shape[0], A.shape[1] * B.T.shape[1]))
    for i in range(T.shape[0]):
        T[i] = np.kron(A[i], B.T[i])
    W = np.linalg.lstsq(T, c)[0]

    # print("rankT", np.linalg.matrix_rank(T))
    # print("rankAug", np.linalg.matrix_rank(np.hstack((T, c))))
    print("zero", T @ W - c)
    W = W.reshape(B.T.shape)
    return W


class FEASE_ONE(UserItemInteractionsAlgorithm):
    def __init__(self, k=2, iterations=10, l2=1):
        super().__init__()
        self.k = k
        self.l2 = l2
        self.iterations = iterations

    def fit(self, X, w=None):
        print("Pre calculate constants")
        l2_k = self.l2 * np.identity(self.k)
        # A = (X.T @ diag_XXT_inv(X) @ diag_XXT_inv(X) @ X + self.l2 * np.identity(X.shape[1]))#.toarray()
        A = (X.T @ diag_XXT_inv(X) @ diag_XXT_inv(X) @ X).toarray()
        # print("A", A)
        G = X.T @ diag_XXT_inv(X) @ X
        Ainv = np.linalg.pinv(A)

        Q = np.random.random((self.k, X.shape[1]))

        for i in range(self.iterations):
            print("Iteration", i)
            QQTinv = np.linalg.inv(Q @ Q.T + l2_k)
            QT_QQTinv = Q.T @ QQTinv
            QT_QQTinv_Q = QT_QQTinv @ Q

            W_term = (np.identity(A.shape[0]) - Ainv @ A)

            T = Ainv * QT_QQTinv_Q
            # print("T", T)
            Tinv = np.linalg.pinv(T)
            TTinv = T @ Tinv
            b = (np.diag(Ainv @ G @ QT_QQTinv_Q) - 1).reshape(A.shape[0], 1)
            # print("b", b)
            c = np.linalg.inv(TTinv - np.identity(A.shape[0])) @ (-TTinv @ b + b)
            # print("c", c)

            # W = solve_AXB_c(W_term, Q, c)
            # w = np.diag(W_term @ W @ Q).reshape(b.shape)
            # print("W", W)

            w = c
            print("w", w)

            bb = b + w
            Dp = np.linalg.pinv(T) @ bb
            print("Dp", Dp)
            print("zero", T @ Dp - bb)

            P = Ainv @ (G - Dp) @ QT_QQTinv + W_term @ W
            # print("P", P)
            print("diag", np.diag(P @ Q))

            PTAPinv = np.linalg.inv(P.T @ A @ P + l2_k)
            P_PTAPinv_PT = P @ PTAPinv @ P.T
            Dq = (1 / np.diag(P_PTAPinv_PT)) * (np.diag(P_PTAPinv_PT @ G) - 1)
            # print("Dq", Dq)
            Q = PTAPinv @ P.T @ (G - np.diag(Dq))
            # print("Q", Q)

            B = P @ Q
            # print("B", B)

            if hasattr(self, "B_"):
                change = np.sum(np.abs(self.B_ - B))
                print("change", change)

            self.B_ = B
            self.P_ = P
            self.Q_ = Q
            
            print()

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        X = diag_XXT_inv(X) @ X
        scores = X @ self.B_
        return scores
