from collections import Counter

import numpy as np
import itertools
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
        raise NotImplementedError()

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
        nonzero_users = list(set(X_pred.nonzero()[0]))
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

    def transform(self, X):
        raise NotImplementedError()

    def sum_per_user(self, X):
        """ Sum of individual scores per user """
        return X.sum(axis=1)

    def ideal_per_user(self, X):
        """ Ideal score per user
        Default: amount of entries in row (for average calculation) """
        return self.count_per_user(X)

    def count_per_user(self, X):
        """ Sum of how many times a value occurs per user (for statistical significance/coverage/...) """
        # empty = nan
        # zeros should be recorded in matrix
        rows, cols = X.indices
        res = np.zeros(X.shape[0])
        for k, v in Counter(rows).items():
            res[k] = v
        return

    def per_user(self, X):
        """ Score per user """
        return self.sum_per_user(X) / self.ideal_per_user(X)

    def per_user_average(self, X):
        """ Average of per user scores """
        l = self.per_user(X)
        return sum(l) / len(l)

    def sum_per_item(self, X):
        """ Sum of individual scores per user """
        return X.sum(axis=0)

    def ideal_per_item(self, X):
        """ Ideal score per item
        Default: amount of entries in row (for average calculation) """
        return self.count_per_item(X)

    def count_per_item(self, X):
        """ Sum of how many times a value occurs per item (for statistical significance/coverage/...) """
        # empty = nan
        # zeros should be recorded in matrix
        rows, cols = X.indices
        res = np.zeros(X.shape[1])
        for k, v in Counter(cols).items():
            res[k] = v
        return

    def per_item(self, X):
        """ Score per item """
        return self.sum_per_item(X) / self.ideal_per_item(X)

    def per_item_average(self, X):
        """ Average of per item scores """
        l = self.per_item(X)
        return sum(l) / len(l)

    def calculate(self, X, y):
        """ Returns Dataframe per user, Dataframe per item and avg per user score. """
        pass
        # TODO: Finish

# TODO: Finish transform for existing metrics


class RecallK(MetricK):
    def __init__(self, K):
        super().__init__(K)
        self.recall = 0
        self.num_users = 0

    def transform(self, X):
        X_pred_top_K = self.get_topK(X)
        result = scipy.sparse.coo_matrix(X.shape)
        # TODO: finish implementing

    def update(self, X_pred, X_true):
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = set(X_pred_top_K[u, :].nonzero()[1])
            true_items = set(X_true[u, :].nonzero()[1])

            self.recall += len(recommended_items.intersection(true_items)) / min(
                self.K, len(true_items)
            )
            self.num_users += 1

        logger.debug(f"Metric {self.name} updated")

        return

    @property
    def name(self):
        return f"Recall_{self.K}"

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.recall / self.num_users


class MeanReciprocalRankK(MetricK):
    def __init__(self, K):
        """
        The MeanReciprocalRank metric reports the first rank at which the prediction
        matches one of the true items for that user.
        """
        super().__init__(K)
        self.rr = 0
        self.num_users = 0

    def update(self, X_pred, X_true):
        # Per user get a sorted list of the topK predicted items
        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred.nonzero()[0]))

        for u in nonzero_users:
            true_items = set(X_true[u, :].nonzero()[1])
            recommended_items = X_pred_top_K[u, :].nonzero()[1]
            item_scores = X_pred_top_K[u, recommended_items].toarray()[0]

            arr_indices = np.argsort(item_scores)

            for ix, item in enumerate(reversed(recommended_items[arr_indices])):
                if item in true_items:
                    self.rr += 1 / (ix + 1)
                    break

            self.num_users += 1

        logger.debug(f"Metric {self.name} updated")

        return

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.rr / self.num_users

    @property
    def name(self):
        return f"MRR_{self.K}"


class NDCGK(MetricK):
    def __init__(self, K):
        super().__init__(K)
        self.NDCG = 0
        self.num_users = 0

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))

        # Calculate IDCG values by creating a list of partial sums (the functional way)
        self.IDCG_cache = [0] + list(
            itertools.accumulate(self.discount_template, lambda x, y: x + y)
        )

    def update(self, X_pred, X_true):

        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred.nonzero()[0]))

        for u in nonzero_users:
            true_items = set(X_true[u, :].nonzero()[1])
            recommended_items = X_pred_top_K[u, :].nonzero()[1]
            item_scores = X_pred_top_K[u, recommended_items].toarray()[0]

            arr_indices = np.argsort(item_scores)

            M = min(self.K, len(true_items))
            if M == 0:
                continue

            # Score of optimal ordering for M true items.
            IDCG = self.IDCG_cache[M]

            # Compute DCG
            DCG = sum(
                (
                    self.discount_template[rank] * (item in true_items)
                    for rank, item in enumerate(
                        reversed(recommended_items[arr_indices])
                    )
                )
            )


            self.num_users += 1
            self.NDCG += DCG / IDCG

        logger.debug(f"Metric {self.name} updated")

        return

    @property
    def name(self):
        return f"NDCG_{self.K}"

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.NDCG / self.num_users


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
