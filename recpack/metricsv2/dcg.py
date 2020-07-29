from recpack.metricsv2.metric import ElementwiseMetric, ListwiseMetric, MetricK

import numpy as np
import pandas as pd

from recpack.utils import logger

# TODO optimizations
# TODO: results in a matrix rather than a list with dict?

class DCG(ElementwiseMetric):
    def __init__(self):
        self.results_per_element = []

        self.discount_template = lambda x: 1.0 / np.log2(x+2) # rank counter starts at 0

    @property
    def name(self):
        return "DCG"

    def update(self, X_pred, X_true):
        nonzero_users = list(set(X_pred.nonzero()[0]))

        for u in nonzero_users:
            true_items = set(X_true[u, :].nonzero()[1])
            recommended_items = X_pred[u, :].nonzero()[1]
            item_scores = X_pred[u, recommended_items].toarray()[0]

            arr_indices = np.argsort(item_scores)

            M = len(true_items)

            if M == 0:
                continue
            
            for rank, item in enumerate(reversed(recommended_items[arr_indices])):
                # TODO: if item not in true items -> do we need to add it to the results?
                DCG = self.discount_template(rank) * (item in true_items)
                self.results_per_element.append({"user": u, "item": item, "dcg": DCG, "rank": rank})

    @property
    def value(self):
        # return average summed 
        return self.results.groupby("user").sum().dcg.mean()

    @property
    def results(self):
        return pd.DataFrame.from_records(self.results_per_element)


class DCGK(DCG, MetricK):
    def __init__(self, K):
        DCG.__init__(self)
        MetricK.__init__(self, K)

    @property
    def name(self):
        return f"DCG_{self.K}"

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
            
            for rank, item in enumerate(reversed(recommended_items[arr_indices])):
                # TODO: if item not in true items -> do we need to add it to the results?
                DCG = self.discount_template(rank) * (item in true_items)
                self.results_per_element.append({"user": u, "item": item, "dcg": DCG, "rank": rank})

# TODO implement NDCG
# ? does it start from DCG? 
#   -> Problem is that it is then both an elementwise and a listwise metric
# Maybe give it a DCG member? and then IDCG template func, 
# so that we can get the results and normalize them?
# At that point it's almost easier to just implement it.
class NDCG(ListwiseMetric):
    pass
