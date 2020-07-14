import logging

import numpy as np
import scipy.sparse

from recpack.metrics import MetricK, MetricResult
from recpack.metrics.ndcg import scores_to_ranks

logger = logging.getLogger("recpack")


class MeanReciprocalRankK(MetricK):
    def __init__(self, K):
        """
        The MeanReciprocalRank metric reports the first rank at which the prediction
        matches one of the true items for that user.
        """
        super().__init__(K)
        self.rr = 0
        self.num_users = 0

    class MRRMetricResult(MetricResult):
        def ideal_per_user(self):
            ideal = self.count_per_user()
            ideal[ideal > 0] = 1
            return ideal

    def transform(self, X):
        # TODO: argpartition can be fastser by exploiting sparse structure of X
        indices = np.argpartition(X.A, -self.K, axis=1)[:, -self.K:]

        # ranks offset by -K (best rank is -K, worst rank is -1)
        allranks = scores_to_ranks(X, indices, offset=-self.K)

        y_mask = self.y.astype(np.bool)
        # print("allranks", allranks)

        filtered_ranks = allranks.multiply(y_mask)
        # print("filtered", filtered_ranks)

        best_ranks_indices = filtered_ranks.argmin(axis=1)
        # print("indices", best_ranks_indices)

        # best ranks (0 means no hit found)
        best_ranks = np.take_along_axis(filtered_ranks, best_ranks_indices, axis=1)
        # print("best", best_ranks.A)

        mask = scipy.sparse.lil_matrix(y_mask.shape, dtype=np.bool)
        np.put_along_axis(mask, best_ranks_indices, best_ranks.astype(np.bool), axis=1)
        mask = mask.tocsr()
        # print("mask", mask)

        scores = scipy.sparse.lil_matrix(X.shape)
        scores[y_mask] = np.nan
        scores[mask] = 1 / (filtered_ranks[mask] + self.K + 1)

        scores = scores.tocsr()
        scores.data = np.nan_to_num(scores.data, nan=0, copy=False)
        # print("scores", scores)

        return MeanReciprocalRankK.MRRMetricResult(scores, self)

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
