import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK

logger = logging.getLogger("recpack")


class AUCAMAN(ElementwiseMetricK):
    """
    Measure is described in paper of Koen Verstrepen and Bart Goethals: Unifying nearest neighbors collaborative
    filtering (10.1145/2645710.2645731).
    It is an AUC measure where a missing preference is evaluated as a dislike (AMAN stands for All Missing As Negative);
    given by the following formula:
        AUC_AMAN = 1 / |U_t| \\sum_{u \\in U_t}{\\frac{|I| - rank(h_u)}{|I| - 1}}
        where H_u is the set containing all test preferences of that user
    """

    def __init__(self, K=None):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """
        Calculate the AUC AMAN score for the particular y_true and y_pred matrices.
        :param y_true: User-item matrix with the actual true rating values.
        :param y_pred: User-item matrix with all prediction rating scores.
        :return: None: The result will be saved in self.value.
        """
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        _, num_items = y_true.shape
        if self.K is None: self.K = num_items
        assert num_items > 1

        y_pred_top_K = self.get_top_K_ranks(y_pred)
        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        scores = scores.multiply(y_pred_top_K)
        scores.data = (num_items - scores.data) / (num_items - 1)

        scores = scores.tocsr()

        self.scores_ = scores
        self.value_ = scores.data.mean()

        return
