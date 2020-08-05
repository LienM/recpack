import torch
import numpy as np

from recpack.metrics import NDCGK

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


class StoppingCriterion(NDCGK):
    def __init__(self, K):
        """
        Stopping Criterion is (should be) a generic wrapper around a Metric,
        allowing resetting of the metric to zero and keeping track of the
        best value.
        """
        # TODO You can do better than this
        # Maybe something with an "initialize" and "refresh" in the actual metrics?
        super().__init__(K)
        self.best_value = -np.inf

    def reset(self):
        """
        Reset the metric
        """
        # TODO Set these in an initialize method for the metric
        # so that Stopping Criterion can call that to reset the metric
        # Then stopping criterion can inherit from metric. BOOM!
        if self.is_best:
            self.best_value = self.value
        self.value_ = 0
        self.num_users_ = 0

    @property
    def is_best(self):
        """
        Is the current value of the metric the best value?
        """
        return True if self.value > self.best_value else False
