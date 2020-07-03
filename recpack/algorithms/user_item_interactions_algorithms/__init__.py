import scipy.sparse

from recpack.algorithms.algorithm_base import Algorithm

from sklearn.utils.validation import check_is_fitted


class UserItemInteractionsAlgorithm(Algorithm):

    def fit(self, X: scipy.sparse.csr_matrix, y: scipy.sparse.csr_matrix=None):
        pass


class SimilarityMatrixAlgorithm(UserItemInteractionsAlgorithm):
    def get_sim_matrix(self):
        raise NotImplementedError()

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        B = self.get_sim_matrix()
        scores = X @ B

        return scores