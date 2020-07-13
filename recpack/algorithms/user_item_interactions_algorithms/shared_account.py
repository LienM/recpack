import scipy.sparse
import numpy as np

import numba

from recpack.algorithms.user_item_interactions_algorithms import (
    SimilarityMatrixAlgorithm,
)


class SharedAccount(SimilarityMatrixAlgorithm):
    def __init__(self, algo: SimilarityMatrixAlgorithm, p=0.75, sum=False, adjustment=False):
        super().__init__()
        self.algo = algo
        self.p = p
        self.sum = sum
        self.adjustment = adjustment
        if sum and adjustment:
            raise RuntimeError("Can't do sum and adjusted average")

    def fit(self, X: scipy.sparse.csr_matrix, y: scipy.sparse.csr_matrix=None):
        return self.algo.fit(X, y)

    def get_sim_matrix(self):
        return self.algo.get_sim_matrix()

    def predict(self, X, user_ids=None):
        M = self.get_sim_matrix().toarray()

        X = X.toarray()
        predictions = get_predictions(X, M, self.p, self.sum, self.adjustment)

        return predictions


@numba.njit(parallel=True)
def get_predictions(X, M, p, sum=False, adjustment=True):
    predictions = np.zeros(X.shape, dtype=np.float32)
    for u in numba.prange(X.shape[0]):
        indices = X[u]
        similarities = M[indices.astype(np.bool_), :]
        predictions[u] = get_prediction_u(similarities, p, sum=sum, adjustment=adjustment)

    return predictions


@numba.njit()
def get_prediction_u(similarities, p, sum=False, adjustment=False):
    predictions = np.zeros((similarities.shape[1]), dtype=np.float32)
    filtered = filter_best_subsets(similarities, p)

    for col in range(filtered.shape[1]):
        nonzero = np.count_nonzero(filtered[:, col])
        # print("set size:", nonzero)
        if nonzero == 0:
            predictions[col] = 0
        elif sum:
            # sum
            predictions[col] = np.sum(filtered[:, col])
        elif adjustment:
            # adjusted average
            predictions[col] = np.sum(filtered[:, col]) / nonzero ** p
        else:
            # average
            predictions[col] = np.sum(filtered[:, col]) / nonzero

    return predictions


@numba.njit()
def filter_best_subsets(similarities, p):
    sort_indices = np.empty(similarities.shape, dtype=np.int32)
    for j in range(sort_indices.shape[1]):
        sort_indices[:, j] = np.argsort(-similarities[:, j])

    for col in range(sort_indices.shape[1]):
        order = sort_indices[:,col]
        total = 0
        amount = 0
        for index in order:
            tmp = (total + similarities[index, col]) / (amount + 1) ** p
            if tmp < total:
                # print("ignore", len(order) - amount, "/", len(order))
                break
            else:
                total = tmp
                amount += 1

        similarities[order[amount:], col] = 0

    return similarities

