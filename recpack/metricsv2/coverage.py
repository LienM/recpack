from scipy.sparse import csr_matrix
from recpack.metricsv2.base import MetricTopK, GlobalMetric, FittedMetric

class Coverage(GlobalMetric, FittedMetric):
    def __init__(self):
        self.number_of_items = 0
        self.covered_items = set()
        
    def fit(self, X):
        # items are assumed to be a continuous sequence of ids.
        # TODO: we could do a nonzero here on the items as well, then the coverage is computed compared to the visited items.
        # But then we also need handling of recommending unvisited items (corner case that should not happen often.)
        self.number_of_items = X.shape[1]

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        # add all nonzero items are covered in this prediction
        self.covered_items = self.covered_items.union(set(y_pred.nonzero()[1]))

        return

    @property
    def value(self):
        if self.number_of_items == 0:
            return 0
        return len(self.covered_items) / self.number_of_items

class CoverageK(Coverage, MetricTopK):
    def __init__(self, K):
        Coverage.__init__(self)
        MetricTopK.__init__(self, K)
    
    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        top_k = self.get_top_K_ranks(y_pred)
        # add all nonzero items are covered in this prediction
        self.covered_items = self.covered_items.union(set(top_k.nonzero()[1]))
        return
