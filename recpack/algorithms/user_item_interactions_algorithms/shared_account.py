import scipy.sparse
import numpy as np

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
        M = self.get_sim_matrix()
        predictions = np.zeros(X.shape, dtype=np.float32)

        for u in range(X.shape[0]):
            print(X[u].toarray())
            indices = X[u].indices
            similarities = M[indices, :].toarray()
            filtered = filter_best_subsets(similarities, self.p)
            # print("filtered", filtered)
            for col in range(filtered.shape[1]):
                nonzero = np.count_nonzero(filtered[:,col])
                print("set size:", nonzero)
                if nonzero == 0:
                    predictions[u, col] = 0
                else:
                    predictions[u, col] = np.sum(filtered[:,col]) / nonzero

        return predictions


def filter_best_subsets(similarities, p):
    sort_indices = np.argsort(-similarities, axis=0)
    for col in range(sort_indices.shape[1]):
        order = sort_indices[:,col]
        total = 0
        amount = 0
        for index in order:
            tmp = (total + similarities[index, col]) / (amount + 1) ** p
            if tmp < total:
                break
            else:
                total = tmp
                amount += 1

        similarities[order[amount:], col] = 0
        # print("highest:", total)
        # print("size:", amount)

    return similarities

