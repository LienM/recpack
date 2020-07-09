import numpy as np
import scipy.sparse

from sklearn.base import TransformerMixin

import logging

logger = logging.getLogger("recpack")


class Metric(TransformerMixin):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return ""

    def fit(self, X, y=None):
        """ X are predictions, y ground truth """
        return self

    def transform(self, X):
        """ X are predictions """
        raise NotImplementedError()


class MetricK(Metric):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self.__class__.__name__ + "_" + str(self.K)

    def get_topK(self, X_pred: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        # Get nonzero users
        nonzero_users = sorted(list(set(X_pred.nonzero()[0])))
        X = X_pred[nonzero_users, :].toarray()

        items = np.argpartition(X, -self.K)[:, -self.K :]

        U, I, V = [], [], []

        for ix, user in enumerate(nonzero_users):
            U.extend([user] * self.K)
            I.extend(items[ix])
            V.extend(X[ix, items[ix]])

        X_pred_top_K = scipy.sparse.csr_matrix(
            (V, (U, I)), dtype=X_pred.dtype, shape=X_pred.shape
        )

        return X_pred_top_K

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X):
        raise NotImplementedError()


class MetricResult(object):
    """
    Wrapper for score per interaction based on a metric.
    Metric calculates score matrix and these functions can aggregate per user, per item and average over all users.
    Extend this class for metric specific behaviour.
    """
    def __init__(self, scores, metric):
        super().__init__()
        self.scores = scores
        self.metric = metric

    @property
    def K(self):
        return self.metric.K

    def sum_per_user(self):
        """ Sum of individual scores per user """
        return self.scores.sum(axis=1).A1

    def ideal_per_user(self):
        """ Ideal score per user
        Default: amount of entries in row, cutoff at K """
        ideal = self.count_per_user()
        ideal[ideal > self.K] = self.K
        return ideal

    def count_per_user(self):
        """ Sum of how many times a value occurs per user (for statistical significance/coverage/...) """
        # empty = nan
        # zeros should be recorded in matrix
        # Number of stored values, including explicit zeros.
        return self.scores.getnnz(axis=1)

    def per_user(self):
        """ Score per user """
        return self.sum_per_user() / self.ideal_per_user()

    def per_user_average(self):
        """ Average of per user scores """
        l = self.per_user()
        l = l[~np.isnan(l)]
        return sum(l) / len(l)

    def sum_per_item(self):
        """ Sum of individual scores per user """
        return self.scores.sum(axis=0)

    def ideal_per_item(self,):
        """ Ideal score per item
        Default: amount of entries in column """
        return self.count_per_item()

    def count_per_item(self):
        """ Sum of how many times a value occurs per item (for statistical significance/coverage/...) """
        # empty = nan
        # zeros should be recorded in matrix
        # Number of stored values, including explicit zeros.
        return self.scores.getnnz(axis=0)

    def per_item(self):
        """ Score per item """
        return self.sum_per_item() / self.ideal_per_item()

    def per_item_average(self):
        """ Average of per item scores """
        l = self.per_item()
        l = l[~np.isnan(l)]
        return sum(l) / len(l)

    def calculate(self):
        """ Returns Dataframe per user, Dataframe per item and avg per user score. """
        pass
        # TODO: Finish
        # TODO: Different name
        # TODO: what about coverage (only per item)


class Coverage(Metric):
    def __init__(self):
        pass


class MutexMetric:
    # TODO Refactor into a Precision metric, a FP and a TN metric.
    """
    Metric used to evaluate the mutex predictors

    Computes false positives as predictor says it's mutex, but sample shows that the item predicted as mutex has been purchased in the evaluation data
    Computes True negatives as items predicted as non mutex, which are also actually in the evaluation data.
    """

    def __init__(self):
        self.false_positives = 0
        self.positives = 0

        self.true_negatives = 0
        self.negatives = 0

    def update(
        self, X_pred: np.matrix, X_true: scipy.sparse.csr_matrix, users: list
    ) -> None:

        self.positives += X_pred.sum()
        self.negatives += (X_pred.shape[0] * X_pred.shape[1]) - X_pred.sum()

        false_pos = scipy.sparse.csr_matrix(X_pred).multiply(X_true)
        self.false_positives += false_pos.sum()

        negatives = np.ones(X_pred.shape) - X_pred
        true_neg = scipy.sparse.csr_matrix(negatives).multiply(X_true)
        self.true_negatives += true_neg.sum()

    @property
    def value(self):
        return (
            self.false_positives,
            self.positives,
            self.true_negatives,
            self.negatives,
        )
