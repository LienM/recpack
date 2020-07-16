import logging
import numpy as np
import scipy.sparse

from recpack.metrics import MetricK, MetricResult


logger = logging.getLogger("recpack")


class RecallK(MetricK):
    def __init__(self, K):
        super().__init__(K)
        self.recall = 0
        self.num_users = 0

    def transform(self, X):
        # TODO: argpartition can be fastser by exploiting sparse structure of X
        indices = np.argpartition(X.A, -self.K, axis=1)[:, -self.K:]
        recommended = scipy.sparse.lil_matrix(X.shape, dtype=np.int32)
        np.put_along_axis(recommended, indices, 1, axis=1)
        recommended = recommended.tocsr()

        scores = scipy.sparse.lil_matrix(self.y.shape)
        # explicitly store zero entries as nan
        scores[self.y.astype(np.bool)] = np.nan
        scores[recommended.multiply(self.y).astype(np.bool)] = 1

        scores = scores.tocsr()
        # convert nan entries back to zero (beacuse csr supports explicit zeros)
        scores.data = np.nan_to_num(scores.data, nan=0)

        return MetricResult(scores, self)

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
