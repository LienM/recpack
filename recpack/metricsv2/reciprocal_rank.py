from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from recpack.metricsv2.base import ListwiseMetric

class RR(ListwiseMetric):
    def __init__(self):
        self.results_per_list = []

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        # TODO: optimize this procedure
        nonzero_users = list(set(y_pred.nonzero()[0]))

        for u in nonzero_users:
            true_items = set(y_true[u, :].nonzero()[1])
            recommended_items = y_pred[u, :].nonzero()[1]
            item_scores = y_pred[u, recommended_items].toarray()[0]

            arr_indices = np.argsort(item_scores)

            self.results_per_list.append({"user": u, "rr": 0})
            for ix, item in enumerate(reversed(recommended_items[arr_indices])):
                if item in true_items:
                    # Update the last entry in the results per list
                    self.results_per_list[-1]["rr"] = 1 / (ix + 1)
                    break

    @property
    def value(self) -> float:
        # This gives MRR as value
        if len(self.results_per_list) == 0:
            return 0
        return sum([x['rr'] for x in self.results_per_list]) / len(self.results_per_list)

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.results_per_list)


# TODO: does a RR@K make sense?