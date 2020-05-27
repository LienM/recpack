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
    def __init__(self, k=10, iterations=10, l2=500):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.iterations = iterations

    def fit(self, X, w=None):

        print("Calculate XTX")
        l2_n = self.l2 * np.identity((X.shape[1]))
        xtx = (X.T @ X).todense()
        print("Invert XTX")
        xtx_inv = np.linalg.inv(xtx + l2_n)

        # Q = np.ones((self.k, X.shape[1]))
        # for row in range(Q.shape[0]):
        #     Q[row] = 10 ** row
        print("Init Q")
        Q = np.random.random((self.k, X.shape[1]))

        # print(Q)

        print("Start iterations")
        for i in range(self.iterations):
            print("Iteration", i)
            P = xtx_inv @ xtx @ Q.T @ np.linalg.pinv(Q @ Q.T)
            Q = np.linalg.inv(P.T @ xtx @ P + self.l2 * P.T @ P) @ P.T @ xtx
            B = P @ Q

            if hasattr(self, "B_"):
                change = np.sum(np.abs(self.B_ - B))
                print("change", change)

            self.B_ = B
            # print("P", P)
            # print("Q", Q)
            # print("B", self.B_)
            # print()

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores


class FEASE2(UserItemInteractionsAlgorithm):
    def __init__(self, k=2, iterations=10, l2=1):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.iterations = iterations

    def fit(self, X, w=None):

        print("Calculate XTX")
        l2_n = self.l2 * np.identity((X.shape[1]))
        l2_k = self.l2 * np.identity((self.k))
        xtx = (X.T @ X).todense()
        xtx_minl2 = xtx - l2_n

        print("Init Q")
        Q = np.random.random((self.k, X.shape[1]))

        print("Start iterations")
        i_knl2 = self.l2 * np.identity((X.shape[1] * self.k))
        for i in range(self.iterations):
            print("Iteration", i)
            A = np.kron(Q@Q.T, xtx) + i_knl2
            c = (xtx_minl2 @ Q.T).flatten('F').T
            p = np.linalg.solve(A, c)
            # print("zero", A @ p - c)
            P = p.T.reshape(Q.T.shape, order='F')
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


class FEASEInv(UserItemInteractionsAlgorithm):
    def __init__(self, k=2, iterations=10, l2=1):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.iterations = iterations

    def fit(self, X, w=None):
        # doesn't converge
        print("Calculate XTX")
        l2_n = self.l2 * np.identity((X.shape[1]))
        l2_n_k = l2_n * self.k
        xtx = X.T @ X
        print("Invert XTX")
        xtx_inv = np.linalg.inv(xtx + l2_n)

        print("Init Q")
        Q = np.random.random((self.k, X.shape[1]))

        print("Start iterations")
        for i in range(self.iterations):
            P = xtx_inv @ ((self.k+1) * xtx + l2_n_k) @ Q.T @ np.linalg.inv(Q @ Q.T)
            Q = np.reciprocal(P).T
            B = P @ Q - self.k * np.identity((X.shape[1]))

            if hasattr(self, "B_"):
                change = np.sum(np.abs(self.B_ - B))
                print("change", change)

            self.B_ = B
            self.P_ = P
            self.Q_ = Q

            # print("P", P)
            # print("Q", Q)
            # print("B", self.B_)
            # print()

        return self

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.B_

        return scores