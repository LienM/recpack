import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from sklearn.base import BaseEstimator

import logging

logger = logging.getLogger("recpack")


class Metric:

    @property
    def name(self):
        return str(self.__class__).lower()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        pass

    @property
    def results(self) -> pd.DataFrame:
        pass

    @property
    def value(self) -> float:
        pass


class ElementwiseMetric(Metric):
    """
    Base class for all metrics that can be calculated for
    each user-item pair.

    Examples are: DCG, Recall/HR
    """
    pass


class ListwiseMetric(Metric):
    """
    Base class for all metrics that can only be calculated
    at the list-level, i.e. one value for each user.

    Examples are: Diversity, nDCG, RR.
    """
    pass


class GlobalMetric(Metric):
    """
    Base class for all metrics that can only be calculated
    as a global number across all items and users.

    Examples are: Coverage.
    """
    pass


class MetricTopK(Metric):
    """
    Base class for any metric computed on the TopK items for a user.
    """
    def __init__(self, K):
        super().__init__()
        self.K = K

    @property
    def name(self):
        return f"{super().name}_{self.K}"


class FittedMetric(Metric, BaseEstimator):
    """
    Base class for all metrics that need to be fit on a training set
    before they can be used.
    """
    def fit(self, X: csr_matrix):
        pass
