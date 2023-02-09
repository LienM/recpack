# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

import torch

from recpack.algorithms import Algorithm
from recpack.algorithms.util import naive_sparse2tensor, get_batches, get_users
from recpack.matrix import to_binary

logger = logging.getLogger("recpack")


class WeightedMatrixFactorization(Algorithm):
    """WMF Algorithm by Yifan Hu, Yehuda Koren and Chris Volinsky et al.

    As described in Hu, Yifan, Yehuda Koren, and Chris Volinsky.
    "Collaborative filtering for implicit feedback datasets."
    2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008

    Based on the input data a confidence of the interaction is computed.
    Parametrized by alpha and epsilon (hyper parameters)

    - If the chosen confidence scheme is ``'minimal'``,
      confidence is computed as ``c(u,i) = 1 + alpha * r(u,i)``.
    - If the chosen confidence scheme is ``'log-scaling'``,
      confidence is computed as ``c(u,i) = 1 + alpha * log(1 + r(u,i)/epsilon)``

    Since the data during fitting is assumed to be implicit,
    this confidence will be the same for all interactions,
    and as such leaving the HP to the defaults works well enough.


    :param confidence_scheme: Which confidence scheme should be used
        to calculate the confidence matrix.
        Options are ["minimal", "log-scaling"].
        Defaults to "minimal"
    :type confidence_scheme: string, optional
    :param alpha: Scaling parameter for generating confidences from ratings.
        Defaults to 40.
    :type alpha: int, optional
    :param epsilon: Small value to avoid division by zero,
        used to compute a confidence from a rating.
        Only used in case cs is set to 'log-scaling'
        Defaults to 1e-8
    :type epsilon: float, optional
    :param num_components: Dimension of the embeddings of both user- and item-factors.
        Defaults to 100
    :type num_components: int, optional
    :param regularization: Regularization parameter used to calculate the Least Squares.
        Defaults to 0.01
    :type regularization: float, optional
    :param iterations: Number of iterations to execute the ALS calculations.
        Defaults to 20
    :type iterations: int, optional
    :param batch_size: Number of users/items to process in every mini batch.
        Defaults to 100
    :type batch_size: int, optional
    """

    CONFIDENCE_SCHEMES = ["minimal", "log-scaling"]
    """Allowed values for confidence scheme parameter"""

    def __init__(
        self,
        confidence_scheme: str = "minimal",
        alpha: int = 40,
        epsilon: float = 1e-8,
        num_components: int = 100,
        regularization: float = 0.01,
        iterations: int = 20,
        batch_size: int = 100,
    ):
        """
        Initialize the weighted matrix factorization algorithm
        with confidence generator parameters.
        """
        super().__init__()
        self.confidence_scheme = confidence_scheme
        if confidence_scheme in self.CONFIDENCE_SCHEMES:
            self.confidence_scheme = confidence_scheme
        else:
            raise ValueError("Invalid confidence scheme parameter.")

        self.alpha = alpha
        self.epsilon = epsilon

        self.num_components = num_components
        self.regularization = regularization
        self.iterations = iterations

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.batch_size = batch_size

        self.loss = torch.nn.MSELoss()

    def _fit(self, X: csr_matrix) -> None:
        """Calculate the user- and item-factors which will approximate X
            after applying a dot-product.

        :param X: Sparse user-item matrix which will be used to fit the algorithm.
        :type X: csr_matrix
        """
        self.num_users, self.num_items = X.shape

        # Create a matrix with only nonzero users
        X_nonzero = self._eliminate_empty_users(X)
        num_nonzero_users = X_nonzero.shape[0]

        C = self._generate_confidence(X_nonzero)

        item_factors = (
            torch.rand((self.num_items, self.num_components), dtype=torch.float32, device=self.device) * 0.01
        )

        for i in tqdm(range(self.iterations)):
            # User iteration
            user_factors = self._least_squares(C, item_factors, (num_nonzero_users, self.num_components))

            # Item iteration
            item_factors = self._least_squares(C.T, user_factors, (self.num_items, self.num_components))

            X_pred = user_factors @ item_factors.T
            loss = self.loss(X_pred, naive_sparse2tensor(X_nonzero).to(self.device))
            logger.debug(f"Current MSE Loss: {loss.item()}")

        self.item_factors_ = item_factors
        self.user_factors_ = torch.zeros(self.num_users, self.num_components, device=self.device)
        self.user_factors_[self.user_id_map_, :] = user_factors

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Prediction scores are calculated as the dot-product of
            the recomputed user-factors and the item-factors.

        :param X: Sparse user-item matrix which will be used to do the predictions;
            only for set of users with interactions will recommendations be generated.
        :type X: csr_matrix
        :return: User-item matrix with the prediction scores as values.
        :rtype: csr_matrix
        """
        U_conf = self._generate_confidence(X)
        U_user_factors = self._least_squares(U_conf, self.item_factors_, (self.num_users, self.num_components))

        score_matrix = csr_matrix((U_user_factors @ self.item_factors_.T).detach().cpu().numpy())

        self._check_prediction(score_matrix, X)
        return score_matrix

    def _generate_confidence(self, r: csr_matrix) -> csr_matrix:
        """
        Generate the confidence matrix as described in the paper.
        This can be calculated in different ways:
          - Minimal: c_ui = alpha * r_ui
          - Log scaling: c_ui = alpha * log(1 + r_ui / epsilon)
        NOTE: This implementation deviates from the paper.
        The additional +1 won't be stored to keep the confidence matrix sparse.
        For this reason C-1 will be the result of this function.
        Important is that it will impact the least squares calculation.
        :param r: User-item matrix which the calculations are based on.
        :type r: csr_matrix
        :return: User-item matrix converted with the confidence values.
        :rtype: csr_matrix
        """
        result = csr_matrix(r, copy=True)
        if self.confidence_scheme == "minimal":
            result.data = self.alpha * result.data
        elif self.confidence_scheme == "log-scaling":
            result.data = self.alpha * np.log(1 + result.data / self.epsilon)

        return result

    def _eliminate_empty_users(self, X: csr_matrix) -> csr_matrix:
        nonzero_users = list(set(X.nonzero()[0]))

        self.user_id_map_ = np.array(nonzero_users)

        return X[nonzero_users, :]

    def _least_squares(self, C: csr_matrix, Y: torch.Tensor, other_factor_dim: Tuple[int, int]) -> torch.Tensor:
        """Calculate the one factor matrix based on the confidence matrix and
        the other factor matrix with the least squares algorithm.

        :param C: (Transposed) Confidence matrix.
        :type C: csr_matrix
        :param Y: Factor matrix used to calculate the other factor.
        :type Y: torch.Tensor
        :param other_factor_dim: Dimension of the factor to be calculated.
        :type other_factor_dim: Tuple[int, int]
        :return: Factor matrix calculated using LS.
        :rtype: torch.Tensor
        """
        YtY = Y.T @ Y

        # accumulate YtCxY + regularization * I in A
        # -----------
        # Because of the impact of calculating C-1, instead of C,
        # calculating YtCxY is a bottleneck, so the significant speedup calculations will be used:
        #  YtCxY = YtY + Yt(Cx)Y
        # Left side of the linear equation A will be:
        #  A = YtY + Yt(Cx)Y + regularization * I
        #  For each x, let us define the diagonal n × n matrix Cx where Cx_yy = c_xy

        binary_C = to_binary(C)

        factors = torch.zeros(other_factor_dim, device=self.device)

        for id_batch in get_batches(get_users(C), batch_size=self.batch_size):
            # Create batches of batch_size
            C_diag_batch = torch.Tensor(C[id_batch, :].toarray()).to(self.device)

            # Used in both A and B
            Y_T_diag = Y.T * C_diag_batch.unsqueeze(1)

            # A batch needs to be a tensor.
            A_batch = YtY + Y_T_diag @ Y + self.regularization * torch.eye(self.num_components, device=self.device)

            P_batch = naive_sparse2tensor(binary_C[id_batch, :]).unsqueeze(-1).to(self.device)

            B_batch = (Y.T + Y_T_diag) @ P_batch

            # Accumulate Yt(Cx + I)Px in b
            # Solve the problem with the A_batch, save results.
            # Xu = (YtCxY + regularization * I)^-1 (YtCxPx)
            x_batch = torch.linalg.lstsq(A_batch, B_batch).solution.squeeze(-1)

            factors[id_batch] = x_batch

        return factors
