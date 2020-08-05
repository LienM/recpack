import torch
import numpy as np
from typing import Callable

from recpack.metrics import NDCGK

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


class StoppingCriterion:
    def __init__(self, metric_class: Callable, *args, **kwargs):
        """
        Stopping Criterion is a container which helps store best metric results and computing the metrics
        allowing resetting of the metric to zero and keeping track of the
        best value.

        :param metric_class: the class to use as metric.
        :type metric_class: Callable
        :param args: Arguments for the metric
        :type args: List
        :param kwargs: Key word arguments for the metric
        :type kwargs: Dict
        """
        self.metric_class = metric_class
        self.metric_args = args
        self.metric_kwargs = kwargs

        self.best_value = -np.inf

        self.metric=None
        
        # Init self.metric
        self.create_metric()

    def create_metric(self):
        # (Re) initialise the metric
        self.metric = self.metric_class(*self.metric_args, **self.metric_kwargs)

    def reset(self):
        """
        Reset the metric
        """
        # TODO Set these in an initialize method for the metric
        # so that Stopping Criterion can call that to reset the metric
        # Then stopping criterion can inherit from metric. BOOM!
        if self.is_best:
            self.best_value = self.value
        # Reset the metric
        self.create_metric()

    def calculate(self, y_true, y_pred):
        self.metric.calculate(y_true, y_pred)

    @property
    def value(self):
        return self.metric.value

    @property
    def is_best(self):
        """
        Is the current value of the metric the best value?
        """
        return True if self.value > self.best_value else False
