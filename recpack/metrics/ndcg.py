import logging
import itertools

import numpy as np
import scipy.sparse

from recpack.metrics import MetricK, MetricResult


logger = logging.getLogger("recpack")


def scores_to_ranks(X, indices, offset=0):
    """ Convert array of scores (X) to array of ranks according to score. Only provided indices are ranked. """
    top_K_scores = np.take_along_axis(X, indices, axis=1).A
    indices_order = np.argsort(-top_K_scores, axis=1)
    K = np.max(indices_order) + 1

    ranks = np.empty_like(indices_order)
    np.put_along_axis(ranks, indices_order, np.arange(offset, K + offset), axis=1)

    allranks = scipy.sparse.lil_matrix(X.shape, dtype=np.int32)
    np.put_along_axis(allranks, indices, ranks, axis=1)
    allranks = allranks.tocsr()
    return allranks


class NDCGK(MetricK):
    def __init__(self, K):
        super().__init__(K)
        self.NDCG = 0
        self.num_users = 0

        self.discount_template = 1.0 / np.log2(np.arange(2, K + 2))

        # Calculate IDCG values by creating a list of partial sums (the functional way)
        self.IDCG_cache = np.array([0] + list(
            itertools.accumulate(self.discount_template, lambda x, y: x + y)
        ))

    class NDCGMetricResult(MetricResult):
        def ideal_per_user(self):
            ideal_counts = super().ideal_per_user()
            ideal = self.metric.IDCG_cache[ideal_counts]
            # print("idealT", ideal[DEBUG_USER])
            return ideal

    def transform(self, X):
        # TODO: argpartition can be fastser by exploiting sparse structure of X
        indices = np.argpartition(X.A, -self.K, axis=1)[:, -self.K:]
        allranks = scores_to_ranks(X, indices)

        y_mask = self.y.astype(np.bool)

        # mask of entries in top K and true positives
        mask = scipy.sparse.lil_matrix(y_mask.shape, dtype=np.bool)
        np.put_along_axis(mask, indices, np.take_along_axis(y_mask, indices, axis=1), axis=1)
        mask = mask.tocsr()

        scores = scipy.sparse.lil_matrix(X.shape)
        # explicitly store zero entries as nan
        scores[y_mask] = np.nan
        scores[mask] = self.discount_template[allranks[mask]]

        scores = scores.tocsr()
        # convert nan entries back to zero (beacuse csr supports explicit zeros)
        scores.data = np.nan_to_num(scores.data, nan=0, copy=False)
        # print("scores", scores)

        return NDCGK.NDCGMetricResult(scores, self)

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
