import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from recpack.metrics.base import ElementwiseMetricK


logger = logging.getLogger("recpack")


class AvgReciprocalHitRateK(ElementwiseMetricK):
    """
    Measure is described in paper of M. Deshpande and G. Karypis: Item-based top-n recommendation algorithms. TOIS,
    22(1):143–177, 2004.
    The Average Reciprocal HitRate@K is a measure which extends the HitRate@K by takeing the rank of the test preference
    in the top 10 of a user into account; given by the following formula:
        arhr@k = 1 / |U_t| \\sum_{u \\in U_t}{|H_u \\union top-k(u)| * 1 / rank(h_u)}
        where H_u is the set containing all test preferences of that user
    """

    def __init__(self, K):
        super().__init__(K)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """
        Calculate the Average Reciprocal HitRate@K score for the particular y_true and y_pred matrices.
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
        y_pred_top_K.data = 1.0 / y_pred_top_K.data  # HitRateK times the inverse of its rank
        scores = scores.multiply(y_pred_top_K)

        scores = scores.tocsr()

        self.scores_ = scores / self.K

        return
