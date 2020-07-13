import scipy.sparse

from recpack.algorithms.algorithm_base import Algorithm

from sklearn.utils.validation import check_is_fitted


class UserItemInteractionsAlgorithm(Algorithm):

    def fit(self, X: scipy.sparse.csr_matrix, y: scipy.sparse.csr_matrix=None):
        pass


class SimilarityMatrixAlgorithm(UserItemInteractionsAlgorithm):

    @property
    def sim_matrix(self):
        raise NotImplementedError()

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.sim_matrix

        return scores