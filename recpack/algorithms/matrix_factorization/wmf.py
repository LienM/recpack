import numpy as np
import logging
from recpack.algorithms import Algorithm
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from .utils import nonzeros

logger = logging.getLogger("recpack")


class WeightedMatrixFactorization(Algorithm):
    """
    WMF Algorithm by Yifan Hu, Yehuda Koren and Chris Volinsky et al.
    as described in paper 'Collaborative Filtering for Implicit Feedback Datasets' (ICDM.2008.22)
    """

    def __init__(self, K: int = None, cs: str = "minimal", alpha: int = 40, epsilon: float = 10 ** (-8),
                 num_components: int = 100, regularization: float = 0.01,
                 iterations: int = 20):
        """
        Initialize the weighted matrix factorization algorithm with confidence generator parameters.
        :param K: (optional) Set the top-K for prediction fase. If K is None, complete prediction matrix will be
        returned.
        :param cs: Which confidence scheme should be used to calculate the confidence matrix. Options are ["minimal",
                   "log-scaling"]
        :param alpha: Alpha parameter for generating confidence matrix.
        :param epsilon: Epsilon parameter for generating confidence matrix.
        :param num_components: Dimension of factors used by the user- and item-factors.
        :param regularization: Regularization parameter used to calculate the Least Squares.
        :param iterations: Number of iterations to execute the ALS calculations.
        """
        super().__init__()
        self.K = K
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
        self.users, self.items = X.shape
        self.known_users = set(X.nonzero()[0])
        self.user_factors_, self.item_factors_ = self._alternating_least_squares(X)

        return self

    def predict(self, X: csr_matrix, user_ids: np.array = None) -> csr_matrix:
        """
        For each user predict the K most popular items.
        :param X: Sparse user-item matrix which will be used to do the predictions; only the set of users will be used.
        :param user_ids: Unused parameter.
        :return: User-item matrix with the prediction scores as values.
        """
        check_is_fitted(self)

        U = set(X.nonzero()[0])
        U_conf = self._generate_confidence(X)
        U_user_factors = self._least_squares_users(U_conf, self.item_factors_, U)

        score_list = []
        for u in U:
            user = U_user_factors[u]
            scores = self.item_factors_.dot(user)  # Prediction is dot product of user and item_factors
            scores_user = [
                (u, i, s)
                for i, s in enumerate(scores)
            ]
            topK = sorted(scores_user, reverse=True, key=(lambda x: x[2]))[:self.K]
            score_list += topK

        user_idxs, item_idxs, scores = list(zip(*score_list))
        score_matrix = csr_matrix(
            (scores, (user_idxs, item_idxs)), shape=X.shape
        )

        self._check_prediction(score_matrix, X)
        return score_matrix

    def _generate_confidence(self, r) -> csr_matrix:
        """
        Generate the confidence matrix as described in the paper.
        This can be calculated in different ways:
          - Minimal: c_ui = 1 + \alpha * r_ui
          - Log scaling: c_ui = 1 + \alpha * log(1 + r_ui / \epsilon)
        :param r: User-item matrix which the calculations are based on.
        :return: User-item matrix converted with the confidence values.
        """
        result = csr_matrix(r, copy=True)
        if self.confidence_scheme == "minimal":
            result.data = 1 + self.alpha * result.data
        elif self.confidence_scheme == "log-scaling":
            result.data = 1 + self.alpha * np.log(1 + result.data / self.epsilon)
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
        user_factors = np.random.rand(self.users, self.num_components).astype(np.float32) * 0.01
        item_factors = np.random.rand(self.items, self.num_components).astype(np.float32) * 0.01

        c = self._generate_confidence(X)
        ct = c.T.tocsr()
        for i in tqdm(range(self.iterations)):
            old_uf = np.array(user_factors, copy=True)
            old_if = np.array(item_factors, copy=True)

            user_factors = self._least_squares_users(c, item_factors)
            item_factors = self._least_squares_items(ct, user_factors)

            norm_uf = np.linalg.norm(old_uf - user_factors, 2)
            norm_if = np.linalg.norm(old_if - item_factors, 2)
            logger.debug(
                f"{self.name} - Iteration {i} - L2-norm of diff user_factors: {norm_uf} - L2-norm of diff "
                f"item_factors: {norm_if}")

        return user_factors, item_factors

    def _least_squares_users(self, matrix_c: csr_matrix, item_factors: np.ndarray, users=None) -> np.ndarray:
        """
        Calculate the user_factor based on the confidence matrix and the item_factors with the least squares algorithm.
        @param matrix_c: Confidence matrix
        @param item_factors: Item factor array
        @param users: Optional parameter to choose with user set to be used, default the self.known_users will be used.
        @return: User factor nd-array based on the item factor array and the confidence matrix
        """
        user_factors = np.zeros((self.users, self.num_components))
        YtY = item_factors.T.dot(item_factors)

        users = self.known_users if users is None else users
        for u in users:
            user_factors[u] = self._linear_equation(item_factors, YtY, matrix_c, u)

        return user_factors

    def _least_squares_items(self, matrix_ct: csr_matrix, user_factors: np.ndarray) -> np.ndarray:
        """
        Calculate the item_factor based on the transposed confidence matrix and the user_factors with the least squares
        algorithm.
        @param matrix_ct: Transposed confidence matrix
        @param user_factors: User factor array
        @return: Item factor nd-array based on the user factor array and the confidence matrix
        """
        item_factors = np.zeros((self.items, self.num_components))
        YtY = user_factors.T.dot(user_factors)

        for i in range(self.items):
            item_factors[i] = self._linear_equation(user_factors, YtY, matrix_ct, i)

        return item_factors

    def _linear_equation(self, Y: np.ndarray, YtY: np.ndarray, matrix: csr_matrix, x: int) -> np.ndarray:
        """
        Helper function to compute the linear equation used in the Least Squares calculations.
        @param Y: Input factor array
        @param YtY: Product of Y transpose and Y.
        @param matrix: The (transposed) confidence matrix
        @param x: Calculation for which item/user x
        @return: Solution for the linear equation (YtCxY + regularization * I)^-1 (YtCxPx)
        """
        # accumulate YtCxY + regularization * I in A
        A = YtY + self.regularization * np.eye(self.num_components)

        # accumulate YtCxPx in b
        b = np.zeros(self.num_components)

        for i, confidence in nonzeros(matrix, x):
            factor = Y[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor

        # Xu = (YtCxY + regularization * I)^-1 (YtCxPx)
        return np.linalg.solve(A, b)
