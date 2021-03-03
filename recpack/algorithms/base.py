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

    def _fit(self, X: csr_matrix):
        raise NotImplementedError("Please implement _fit")

    def _predict(self, X: csr_matrix) -> csr_matrix:
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self):
        """Helper function to check that fit stored information."""
        # Use the sklear check_is_fitted function,
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        check_is_fitted(self)

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

    def fit(self, X: Matrix) -> "Algorithm":
        # TODO: There might be a way to use this
        # to the advantage of handling csr_matrix or not
        # Eg. 1 base class uses to_csr, the other an InteractionMatrix?
        self._fit(X)

        self._check_fit_complete()
        return self

    def predict(self, X: Matrix) -> csr_matrix:
        check_is_fitted(self)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred


class SimilarityMatrixAlgorithm(Algorithm):
    def _predict(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        scores = X @ self.similarity_matrix_

        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores

    def _check_fit_complete(self):
        """Checks if the fitted matrix, contains a similarity for each item.
        Uses warnings to push this info to the customer.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar items for {missing} items.")


class TopKSimilarityMatrixAlgorithm(SimilarityMatrixAlgorithm):
    def __init__(self, K):
        super().__init__()
        self.K = K
