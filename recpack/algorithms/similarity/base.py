from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csr_matrix

from recpack.algorithms.base import Algorithm


class SimilarityMatrixAlgorithm(Algorithm):

    def fit(self, X: csr_matrix, y: csr_matrix = None):
        pass

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        scores = X @ self.similarity_matrix_

        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores


class TopKSimilarityMatrixAlgorithm(SimilarityMatrixAlgorithm):

    def __init__(self, K):
        super().__init__()
        self.K = K
