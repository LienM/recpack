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
        # xtx = X.T.dot(X).todense()
        xtx = X.T @ X + self.l2 * np.identity((X.shape[1]), dtype=np.float32)
        print("Invert XTX")
        xtx_inv = np.linalg.inv(xtx)

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

