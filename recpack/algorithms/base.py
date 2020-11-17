import scipy.sparse
import numpy as np
from sklearn.base import BaseEstimator
import warnings
from recpack.data.matrix import Matrix


class Algorithm(BaseEstimator):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = ",".join(
            (f"{k}={v}" for k, v in self.get_params().items()))
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

    def _check_prediction(self, X_pred: scipy.sparse.csr_matrix,
                          X: scipy.sparse.csr_matrix) -> None:
        """Checks that the prediction matches expectations.

        Checks implemented
        - Check that all users with history got at least 1 recommendation

        :param X_pred: The matrix with predictions
        :type X_pred: scipy.sparse.csr_matrix
        :param X: The input matrix for prediction, used as 'history'
        :type X: scipy.sparse.csr_matrix
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warnings.warn(
                f"{self.name} failed to recommend any items "
                f"for {len(missing)} users")
