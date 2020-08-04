from recpack.metricsv2.base import FittedMetric, ListwiseMetric, MetricTopK
from scipy.spatial import distance
from scipy.sparse import csr_matrix
import pandas as pd

class IntraListDiversityK(FittedMetric, ListwiseMetric, MetricTopK):
    def __init__(self, K):
        MetricTopK.__init__(self, K)
        FittedMetric.__init__(self)
        ListwiseMetric.__init__(self)

        self.X = None
        self.results_per_list = []

    def fit(self, X: csr_matrix) -> None:
        """X is an item to feature matrix, with features one hot encoded"""
        self.X = X

    def _get_distance(self, i, j):
        return distance.jaccard(self.X[i].toarray()[0], self.X[j].toarray()[0])

    def _get_ild(self, recommended_items):
        # Compute the ILD for this list
        # Sum part: SUM(d(i_k, i_l) for i_k in R and l < k)
        # We will compute this sum by constructing a sparse matrix with a 1 on each of the required i_k, i_l tuples
        if len(recommended_items) <= 1:
            # If there are 1 or no items, the intra list distance is 0
            return 0
        coordinates = [(i_k, i_l) for i, i_k in enumerate(recommended_items) for i_l in recommended_items[:i]]
        distances = [self._get_distance(i, j) for i,j in coordinates]

        t_distance = sum(distances)

        ild = (2 / (len(recommended_items)*(len(recommended_items)-1))) * t_distance
        return ild

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """ Only looks at the predicted items, does not care about the 'true' items.
        """
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        # resolve top K items per user
        y_pred_top_K = self.get_top_K_ranks(y_pred)

        nonzero_users = list(set(y_pred_top_K.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = list(set(y_pred_top_K[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            self.results_per_list.append({"user": u, "score": self._get_ild(recommended_items)})

        self._value = sum([x["score"] for x in self.results_per_list])/ len(self.results_per_list)

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.results_per_list)
