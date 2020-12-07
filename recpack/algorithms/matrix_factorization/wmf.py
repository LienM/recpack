import logging

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from recpack.algorithms import Algorithm

logger = logging.getLogger("recpack")


class WeightedMatrixFactorization(Algorithm):
    """
    WMF Algorithm by Yifan Hu, Yehuda Koren and Chris Volinsky et al.
    as described in paper 'Collaborative Filtering for Implicit Feedback Datasets' (ICDM.2008.22)
    """

    def __init__(self, cs: str = "minimal", alpha: int = 40, epsilon: float = 10 ** (-8),
                 num_components: int = 100, regularization: float = 0.01,
                 iterations: int = 20):
        """
        Initialize the weighted matrix factorization algorithm with confidence generator parameters.
        :param cs: Which confidence scheme should be used to calculate the confidence matrix. Options are ["minimal",
                   "log-scaling"]
        :param alpha: Alpha parameter for generating confidence matrix.
        :param epsilon: Epsilon parameter for generating confidence matrix.
        :param num_components: Dimension of factors used by the user- and item-factors.
        :param regularization: Regularization parameter used to calculate the Least Squares.
        :param iterations: Number of iterations to execute the ALS calculations.
        """
        super().__init__()
        self.confidence_scheme = cs
        self.alpha = alpha
        self.epsilon = epsilon

        self.num_components = num_components
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, X: csr_matrix) -> Algorithm:
        """
        Calculate the user- and item-factors which will be approximate X after applying a dot-product.
        :param X: Sparse user-item matrix which will be used to fit the algorithm.
        :return: The fitted WeightedMatrixFactorizationAlgorithm itself.
        """
        self.num_users, self.num_items = X.shape
        self.known_users = set(X.nonzero()[0])
        self.user_factors_, self.item_factors_ = self._alternating_least_squares(X)

        return self

    def predict(self, X: csr_matrix, user_ids: np.array = None) -> csr_matrix:
        """
        The prediction can easily be calculated as the dotproduct of the recalculated user-factor and the item-factor.
        :param X: Sparse user-item matrix which will be used to do the predictions; only the set of users will be used.
        :param user_ids: Unused parameter.
        :return: User-item matrix with the prediction scores as values.
        """
        check_is_fitted(self)

        U = set(X.nonzero()[0])
        U_conf = self._generate_confidence(X)
        U_user_factors = self._least_squares(U_conf, self.item_factors_, self.num_users, U)

        score_matrix = csr_matrix(U_user_factors @ self.item_factors_.T)

        self._check_prediction(score_matrix, X)
        return score_matrix

    def _generate_confidence(self, r) -> csr_matrix:
        """
        Generate the confidence matrix as described in the paper.
        This can be calculated in different ways:
          - Minimal: c_ui = \alpha * r_ui
          - Log scaling: c_ui = \alpha * log(1 + r_ui / \epsilon)
        NOTE: This implementation deviates from the paper. The additional +1 won't be stored in memory to keep the
        confidence matrix sparse. For this reason C-1 will be the result of this function. Important is that it will
        infect the least squares calculation.
        :param r: User-item matrix which the calculations are based on.
        :return: User-item matrix converted with the confidence values.
        """
        result = csr_matrix(r, copy=True)
        if self.confidence_scheme == "minimal":
            result.data = self.alpha * result.data
        elif self.confidence_scheme == "log-scaling":
            result.data = self.alpha * np.log(1 + result.data / self.epsilon)
        else:
            raise ValueError("Invalid confidence scheme parameter.")

        return result

    def _alternating_least_squares(self, X: csr_matrix) -> (np.ndarray, np.ndarray):
        """
        The ALS algorithm will execute the least squares calculation for x number of iterations.
        According factorizing matrix C into two factors Users and Items such that R \approx U^T I.
        :param X: Sparse matrix which the ALS algorithm should be applied on.
        :return: Generated user- and item-factors based on the input matrix X.
        """
        user_factors = np.random.rand(self.num_users, self.num_components).astype(np.float32) * 0.01
        item_factors = np.random.rand(self.num_items, self.num_components).astype(np.float32) * 0.01

        c = self._generate_confidence(X)
        ct = c.T.tocsr()
        item_set = set(range(self.num_items))

        for i in tqdm(range(self.iterations)):
            old_uf = np.array(user_factors, copy=True)
            old_if = np.array(item_factors, copy=True)

            user_factors = self._least_squares(c, item_factors, self.num_users, self.known_users)
            item_factors = self._least_squares(ct, user_factors, self.num_items, item_set)

            norm_uf = np.linalg.norm(old_uf - user_factors, 2)
            norm_if = np.linalg.norm(old_if - item_factors, 2)
            logger.debug(
                f"{self.name} - Iteration {i} - L2-norm of diff user_factors: {norm_uf} - L2-norm of diff "
                f"item_factors: {norm_if}")

        return user_factors, item_factors

    def _least_squares(self, conf_matrix: csr_matrix, factors: np.ndarray, dimension: int, distinct_set: set) \
            -> np.ndarray:
        """
        Calculate the other factor based on the confidence matrix and the factors with the least squares algorithm.
        It is a general function for item- and user-factors. Depending on the parameter factor_type the other factor
        will be calculated.
        @param conf_matrix: (Transposed) Confidence matrix
        @param factors: Factor array
        @param dimension: User/item dimension.
        @param distinct_set: Set of users/items
        @return: Other factor nd-array based on the factor array and the confidence matrix
        """
        factors_x = np.zeros((dimension, self.num_components))
        YtY = factors.T @ factors

        for i in distinct_set:
            factors_x[i] = self._linear_equation(factors, YtY, conf_matrix, i)

        return factors_x

    def _linear_equation(self, Y: np.ndarray, YtY: np.ndarray, C: csr_matrix, x: int) -> np.ndarray:
        """
        Helper function to compute the linear equation used in the Least Squares calculations.
        @param Y: Input factor array
        @param YtY: Product of Y transpose and Y.
        @param C: The (transposed) confidence matrix
        @param x: Calculation for which item/user x
        @return: Solution for the linear equation (YtCxY + regularization * I)^-1 (YtCxPx)
        """
        # accumulate YtCxY + regularization * I in A
        # -----------
        # Because of the impact of calculating C-1, instead of C,
        # calculating YtCxY is a bottleneck, so the significant speedup calculations will be used:
        #  YtCxY = YtY + Yt(Cx)Y
        # Left side of the linear equation A will be:
        #  A = YtY + Yt(Cx)Y + regularization * I
        #  For each x, let us define the diagonal n × n matrix Cx where Cx_yy = c_xy
        cx = C[x]
        Cx = diags(C[x].toarray().flatten(), 0)
        A = YtY + (Y.T * Cx) @ Y + self.regularization * np.eye(self.num_components)

        # accumulate Yt(Cx + I)Px in b
        cx[cx > 0] = 1  # now Px is represented as cx
        b = (Y.T * (Cx + eye(Cx.shape[0]))) @ cx.T.toarray()

        # Xu = (YtCxY + regularization * I)^-1 (YtCxPx)
        #  Flatten the result to make sure the dimension is (self.num_components,)
        return np.linalg.solve(A, b).flatten()
