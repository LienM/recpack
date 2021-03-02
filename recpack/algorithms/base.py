from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

import warnings
from recpack.data.matrix import to_csr_matrix, Matrix


class Algorithm(BaseEstimator):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def fit(self, X: Matrix):
        pass

    def predict(self, X: Matrix):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def _check_prediction(self, X_pred: csr_matrix, X: csr_matrix) -> None:
        """Checks that the prediction matches expectations.

        Checks implemented
        - Check that all users with history got at least 1 recommendation

        :param X_pred: The matrix with predictions
        :type X_pred: csr_matrix
        :param X: The input matrix for prediction, used as 'history'
        :type X: csr_matrix
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warnings.warn(
                f"{self.name} failed to recommend any items "
                f"for {len(missing)} users"
            )


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
