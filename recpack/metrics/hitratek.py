import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK


logger = logging.getLogger("recpack")


class HitRateK(ElementwiseMetricK):
    """
    Measure is described in paper of M. Deshpande and G. Karypis: Item-based top-n recommendation algorithms. TOIS,
    22(1):143–177, 2004.
    The HitRate@K is a measure which gives the percentage of test users for which the test preference is in the top K
    recommendations; given by the following formula:
        hr@k = 1 / |U_t| \\sum_{u \\in U_t}{|H_u \\union top-k(u)|}
        where H_u is the set containing all test preferences of that user
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """
        Calculate the HitRate@K score for the particular y_true and y_pred matrices.
        :param y_true: User-item matrix with the actual true rating values.
        :param y_pred: User-item matrix with all prediction rating scores.
        :return: None: The result will be saved in self.value.
        """
        y_true, y_pred = self.eliminate_empty_users(y_true, y_pred)
        self.verify_shape(y_true, y_pred)

        y_pred_top_K = self.get_top_K_ranks(y_pred)

        scores = scipy.sparse.lil_matrix(y_pred.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        scores = scores.tocsr()

        self.scores_ = scores
        sums = scores.sum(axis=1) / self.K
        self.value_ = sums.mean()
        return
