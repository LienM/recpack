from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csr_matrix
import warnings

from recpack.algorithms.base import Algorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class SimilarityMatrixAlgorithm(Algorithm):
    def fit(self, X: Matrix):
        pass

    def predict(self, X: Matrix):
        check_is_fitted(self)
        X = to_csr_matrix(X, binary=True)

        scores = X @ self.similarity_matrix_

        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        self._check_prediction(scores, X)

        return scores

    def _check_fit_complete(self):
        """Checks if the fitted matrix, contains a similarity for each item.
        Uses warnings to push this info to the customer.
        """
        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar items for {missing} items.")


class TopKSimilarityMatrixAlgorithm(SimilarityMatrixAlgorithm):
    def __init__(self, K):
        super().__init__()
        self.K = K
