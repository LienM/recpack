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


def has_solutions(A, b):
    rankA = np.linalg.matrix_rank(A)
    rankAug = np.linalg.matrix_rank(np.hstack((A, b)))
    return rankA >= rankAug


class FEASE_ONE(UserItemInteractionsAlgorithm):
    def __init__(self, k=2, iterations=10, l2_1=0.1, l2_2=1):
        super().__init__()
        self.k = k
        self.l2_1 = l2_1
        self.l2_2 = l2_2
        self.iterations = iterations

    def fit(self, X, w=None):
        print("Pre calculate constants")
        Ik = np.identity(self.k)
        In = np.identity(X.shape[1])

        # A = (X.T @ diag_XXT_inv(X) @ diag_XXT_inv(X) @ X + self.l2 * np.identity(X.shape[1]))#.toarray()
        A = (X.T @ diag_XXT_inv(X) @ diag_XXT_inv(X) @ X).toarray()
        # print("A", A)
        G = X.T @ diag_XXT_inv(X) @ X + self.l2_1 * self.l2_2 * In
        # Ainv = np.linalg.pinv(A)
        Ainv = np.linalg.inv(A + self.l2_1 * In)

        Q = np.random.random((self.k, X.shape[1]))
        print("Qinit", Q)
        # P = np.random.random((self.k, X.shape[1])).T

        for i in range(self.iterations):
            print("Iteration", i)

            QQTinv = np.linalg.inv(Q @ Q.T + self.l2_2 * Ik)
            QT_QQTinv = Q.T @ QQTinv
            QT_QQTinv_Q = QT_QQTinv @ Q

            T = Ainv * QT_QQTinv_Q
            # print("T", T)
            Tinv = np.linalg.pinv(T)
            # TTinv = T @ Tinv
            b = (np.diag(Ainv @ G @ QT_QQTinv_Q) - 1).reshape(A.shape[0], 1)
            Dp = (Tinv @ b).flatten()
            # print("Dp", Dp)
            # print("zero", T @ Dp.reshape(A.shape[0], 1) - b)

            P = Ainv @ (G - np.diag(Dp)) @ QT_QQTinv
            # print("P", P)
            print("diag_P", np.diag(P @ Q))

            PTAPinv = np.linalg.inv(P.T @ (A + self.l2_1 * In) @ P + (self.l2_2 + self.l2_1 * self.l2_2) * Ik)
            P_PTAPinv_PT = P @ PTAPinv @ P.T

            Dq = (1 / np.diag(P_PTAPinv_PT)) * (np.diag(P_PTAPinv_PT @ G) - 1)
            # print("Dq", Dq)
            Q = PTAPinv @ P.T @ (G - np.diag(Dq))
            # print("Q", Q)
            print("diag_Q", np.diag(P @ Q))

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


class FEASE_ONE_Test(UserItemInteractionsAlgorithm):
    def __init__(self, k=2, iterations=10, l2=1, rho=0.01):
        super().__init__()
        self.k = k
        self.l2 = l2
        self.iterations = iterations
        self.rho = rho

    def fit(self, X, w=None):
        print("Pre calculate constants")
        l2_k = self.l2 * np.identity(self.k)
        # A = (X.T @ diag_XXT_inv(X) @ diag_XXT_inv(X) @ X + self.l2 * np.identity(X.shape[1]))#.toarray()
        A = (X.T @ diag_XXT_inv(X) @ diag_XXT_inv(X) @ X).toarray()
        # print("A", A)
        G = X.T @ diag_XXT_inv(X) @ X
        Ainv = np.linalg.inv(A + self.l2 * np.identity(X.shape[1]))

        Q = np.random.random((self.k, X.shape[1]))
        D = np.zeros((X.shape[1]))

        for i in range(self.iterations):
            print("Iteration", i)

            # update P
            QQTinv = np.linalg.pinv(Q @ Q.T)
            QT_QQTinv = Q.T @ QQTinv
            P = Ainv @ (G - np.diag(D)) @ QT_QQTinv

            # update Q
            PTAPinv = np.linalg.inv(P.T @ A @ P + l2_k)
            Q = PTAPinv @ P.T @ (G - np.diag(D))

            # working version of B
            B = P @ Q

            # constraint
            D = D + self.rho * (np.diag(B) - 1)

            # TODO: try update P and Q based on two previous versions (not current iteration)

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
