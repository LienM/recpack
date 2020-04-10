import scipy.sparse

from recpack.algorithms.algorithm_base import Algorithm


class UserItemInteractionsAlgorithm(Algorithm):

    def fit(self, X: scipy.sparse.csr_matrix):
        pass
