from recpack.metricsv2.base import ListwiseMetric, MetricTopK
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from recpack.utils import logger
# TODO: optimisations

class Recall(ListwiseMetric):
    def __init__(self):
        ListwiseMetric.__init__(self)

        self.results_per_list = []

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:

        nonzero_users = list(set(y_pred.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = set(y_pred[u, :].nonzero()[1])
            true_items = set(y_true[u, :].nonzero()[1])

            self.results_per_list.append({"user": u, "recall": 0})
            self.results_per_list[-1]["recall"] = len(recommended_items.intersection(true_items)) / len(true_items)

        logger.debug(f"Metric {self.name} updated")

        return

    @property
    def value(self) -> float:
        if len(self.results_per_list) == 0:
            return 0
        return sum([x['recall'] for x in self.results_per_list]) / len(self.results_per_list)

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.results_per_list)


class RecallK(Recall, MetricTopK):
    def __init__(self, K):
        Recall.__init__(self)
        MetricTopK.__init__(self, K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        y_pred_top_K = self.get_topK(y_pred)

        nonzero_users = list(set(y_pred.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = set(y_pred_top_K[u, :].nonzero()[1])
            true_items = set(y_true[u, :].nonzero()[1])

            self.results_per_list.append({"user": u, "recall": 0})
            self.results_per_list[-1]["recall"] =\
                len(recommended_items.intersection(true_items)) / min(self.K, len(true_items))

        logger.debug(f"Metric {self.name} updated")
        return
