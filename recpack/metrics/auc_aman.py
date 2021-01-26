import logging

import numpy as np
import scipy.sparse
from recpack.util import get_top_K_ranks
from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK

logger = logging.getLogger("recpack")

class AUCAMAN(ListwiseMetricK):
    """
    Measure is described in paper of S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. Bpr: Bayesian
    personalized ranking from implicit feedback. In UAI, pages 452–461, 2009
    The measure is implemented as worked out in equation 2.
    """

    def __init__(self):
        super().__init__(None)

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix, X: csr_matrix) -> None:
        """
        Calculate the AUC AMAN score for the particular y_true and y_pred matrices.
        :param y_true: User-item matrix with the actual true binary rating values.
        :param y_pred: User-item matrix with all prediction rating scores.
        :param X: User-item matrix with all binary rating interactions.
        :return: None: The result will be saved in self.value.
        """
        self.verify_shape(y_true, y_pred)

        num_users, num_items = X.shape

        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        # Rank of the matrix I
        ranking = scipy.sparse.lil_matrix(X.shape)

        # Elementwise multiplication of top K predicts and true interactions
        ranking[y_pred_top_K.multiply(y_true).astype(np.bool)] = 1
        ranking = ranking.multiply(y_pred_top_K)
        ranking = ranking.tocsr()  # contains the ranks for the i elements in E(u)

        J = X + y_true  # X and y_true: not in J

        scores = csr_matrix((num_users, 1))

        for u in range(num_users):
            counter = 1
            rank_u = ranking[u, :]
            if rank_u.nnz == 0:  # is empty
                continue
            I_rank = sorted(rank_u.data)

            J_u = J[u, :]
            E_u = num_items - len(J_u.data)
            sum = 0

            # For all i:
            #  SUM (E(u) - rank_i + counter) / E(u)
            for rank in I_rank:
                sum += (E_u - rank + counter) / E_u
                counter += 1

            scores[u, 0] = sum / len(I_rank)  # divide by length of I_rank to scale between 0 and 1

        self.scores_ = scores

        return
