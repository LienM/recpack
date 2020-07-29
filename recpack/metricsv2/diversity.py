from recpack.metricsv2.metric import FittableMetric, ListwiseMetric, MetricK
from scipy.spatial import distance
import pandas as pd


class IntraListDiversity(FittableMetric, ListwiseMetric):
    def __init__(self):
        FittableMetric.__init__(self)
        ListwiseMetric.__init__(self)

        self.X = None
        self.results_per_list = []

    @property
    def name(self):
        return "intra_list_diversity"
    
    def fit(self, X):
        """X is an item to feature matrix, with features one hot encoded"""
        self.X = X

    def _get_distance(self, i, j):
        return distance.jaccard(self.X[i].toarray()[0], self.X[j].toarray()[0])

    def _get_ild(self, recommended_items):
        # Compute the IDL for this list
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

    def update(self, X_pred, X_true):
        """ Only looks at the predicted items, does not care about the 'true' items.
        """
        nonzero_users = list(set(X_pred.nonzero()[0]))
        
        for u in nonzero_users:
            recommended_items = list(set(X_pred[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            self.results_per_list.append({"user": u, "diversity": self._get_ild(recommended_items)})

    @property
    def value(self):
        """Returns the average diversity"""
        if len(self.results_per_list) == 0:
            return 0
        return (sum([x["diversity"] for x in self.results_per_list])/ len(self.results_per_list))
    
    @property
    def results(self):
        return pd.DataFrame.from_records(self.results_per_list)


class IntraListDiversityK(IntraListDiversity, MetricK):
    def __init__(self, K):
        IntraListDiversity.__init__(self)
        MetricK.__init__(self, K)
    
    @property
    def name(self):
        return f"intra_list_diversity_{self.K}"
    
    def update(self, X_pred, X_true):
        """ Only looks at the predicted items, does not care about the 'true' items.
        """
        # resolve top K items per user
        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred_top_K.nonzero()[0]))

        for u in nonzero_users:
            recommended_items = list(set(X_pred_top_K[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            self.results_per_list.append({"user": u, "diversity": self._get_ild(recommended_items)})
