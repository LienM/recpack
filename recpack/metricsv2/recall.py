from recpack.metricsv2.metric import ListwiseMetric, MetricK
import numpy as np
import pandas as pd

from recpack.utils import logger
# TODO: optimisations

class Recall(ListwiseMetric):
    def __init__(self):
        ListwiseMetric.__init__(self)

        self.results_per_list = []

    @property
    def name(self):
        return "recall"

    def update(self, X_pred, X_true):

        nonzero_users = list(set(X_pred.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = set(X_pred[u, :].nonzero()[1])
            true_items = set(X_true[u, :].nonzero()[1])

            self.results_per_list.append({"user": u, "recall": 0})
            self.results_per_list[-1]["recall"] = len(recommended_items.intersection(true_items)) / len(true_items)

        logger.debug(f"Metric {self.name} updated")

        return

    @property
    def value(self):
        if len(self.results_per_list) == 0:
            return 0
        return sum([x['recall'] for x in self.results_per_list]) / len(self.results_per_list)

    @property
    def results(self):
        return pd.DataFrame.from_records(self.results_per_list)


class RecallK(Recall, MetricK):
    def __init__(self, K):
        Recall.__init__(self)
        MetricK.__init__(self, K)

    @property
    def name(self):
        return f"recall_{self.K}"

    def update(self, X_pred, X_true):
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = set(X_pred_top_K[u, :].nonzero()[1])
            true_items = set(X_true[u, :].nonzero()[1])

            self.results_per_list.append({"user": u, "recall": 0})
            self.results_per_list[-1]["recall"] =\
                len(recommended_items.intersection(true_items)) / min(self.K, len(true_items))

        logger.debug(f"Metric {self.name} updated")
        return
