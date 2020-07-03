import scipy.sparse
import numpy as np

from tqdm.auto import tqdm
import numba

from recpack.algorithms.user_item_interactions_algorithms import (
    SimilarityMatrixAlgorithm,
)


class SharedAccount(SimilarityMatrixAlgorithm):
    def __init__(self, algo: SimilarityMatrixAlgorithm, p=0.75):
        super().__init__()
        self.algo = algo
        self.p = p

    def fit(self, X: scipy.sparse.csr_matrix, y: scipy.sparse.csr_matrix=None):
        return self.algo.fit(X, y)

    def get_sim_matrix(self):
        return self.algo.get_sim_matrix()

    def predict(self, X, user_ids=None):
        M = self.get_sim_matrix().toarray()

        X = X.toarray()
        predictions = get_predictions(X, M, self.p)

        return predictions


# @numba.njit()
@numba.njit(parallel=True)
def get_predictions(X, M, p):
    predictions = np.zeros(X.shape, dtype=np.float32)
    for u in numba.prange(X.shape[0]):
        indices = X[u]
        similarities = M[indices.astype(np.bool_), :]
        predictions[u] = get_prediction_u(similarities, p)

    return predictions


# @numba.njit(parallel=True)
@numba.njit()
def get_prediction_u(similarities, p):
    predictions = np.zeros((similarities.shape[1]), dtype=np.float32)
    filtered = filter_best_subsets(similarities, p)

    # print("filtered", filtered)
    for col in range(filtered.shape[1]):
        nonzero = np.count_nonzero(filtered[:, col])
        # print("set size:", nonzero)
        # tqdm.write(f"set size: {nonzero}")
        if nonzero == 0:
            predictions[col] = 0
        else:
            predictions[col] = np.sum(filtered[:, col]) / nonzero
    return predictions


# @numba.njit(parallel=True)
@numba.njit()
def filter_best_subsets(similarities, p):
    # sort_indices = np.argsort(-similarities, axis=0)
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
        # print("highest:", total)
        # print("size:", amount)

    return similarities

