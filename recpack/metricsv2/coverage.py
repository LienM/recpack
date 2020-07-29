import scipy.sparse
from recpack.metricsv2.metric import MetricK, GlobalMetric, FittableMetric

class Coverage(GlobalMetric, FittableMetric):
    def __init__(self):
        self.number_of_items = 0
        self.covered_items = set()
    
    @property
    def name(self):
        return "coverage"
    
    def fit(self, X):
        # items are assumed to be a continuous sequence of ids.
        # TODO: we could do a nonzero here on the items as well, then the coverage is computed compared to the visited items.
        # But then we also need handling of recommending unvisited items (corner case that should not happen often.)
        self.number_of_items = X.shape[1]

    def update(self, X_pred: scipy.sparse.csr_matrix, X_true: scipy.sparse.csr_matrix):
        # add all nonzero items are covered in this prediction
        self.covered_items = self.covered_items.union(set(X_pred.nonzero()[1]))
        return

    @property
    def value(self):
        if self.number_of_items == 0:
            return 0
        return len(self.covered_items) / self.number_of_items

class CoverageK(Coverage, MetricK):
    def __init__(self, K):
        Coverage.__init__(self)
        MetricK.__init__(self, K)

    @property
    def name(self):
        return f"coverage_{self.K}"

    
    def update(self, X_pred: scipy.sparse.csr_matrix, X_true: scipy.sparse.csr_matrix):
        top_k = self.get_topK(X_pred)
        # add all nonzero items are covered in this prediction
        self.covered_items = self.covered_items.union(set(top_k.nonzero()[1]))
        return
