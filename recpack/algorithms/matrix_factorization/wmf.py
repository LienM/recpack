import numpy as np
import logging
from recpack.algorithms import Algorithm
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from .utils import nonzeros

logger = logging.getLogger("recpack")


class WeightedMatrixFactorizationAlgorithm(Algorithm):
    """
    WMF Algorithm by Yifan Hu, Yehuda Koren and Chris Volinsky et al.
    as described in paper 'Collaborative Filtering for Implicit Feedback Datasets' (ICDM.2008.22)
    """

    def __init__(self, K: int = None, cs: str = "minimal", alpha: int = 40, epsilon: float = 10 ** (-8),
                 factors: int = 100, regularization: float = 0.01,
                 iterations: int = 20):
        """
        Initialize the weighted matrix factorization algorithm with confidence generator parameters.
        :param K: (optional) Set the top-K for prediction fase. If K is None, complete prediction matrix will be
        returned.
        :param cs: Which confidence scheme should be used to calculate the confidence matrix. Options are ["minimal",
                   "log-scaling"]
        :param alpha: Alpha parameter for generating confidence matrix.
        :param epsilon: Epsilon parameter for generating confidence matrix.
        :param factors: Dimension of factors used by the user- and item-factors.
        :param regularization: Regularization parameter used to calculate the Least Squares.
        :param iterations: Number of iterations to execute the ALS calculations.
        """
        super().__init__()
        self.K = K
        self.r = None
        self.c = None
        self.confidence_scheme = cs
        self.alpha = alpha
        self.epsilon = epsilon

        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, X: csr_matrix) -> Algorithm:
        """
        Calculate the user- and item-factors which will be approximate X after applying a dot-product.
        :param X: Sparse user-item matrix which will be used to fit the algorithm.
        :return: The fitted WeightedMatrixFactorizationAlgorithm itself.
        """
        self.r = X
        self.c = self._generate_confidence(X)
        self._alternating_least_squares()

        return self

    def predict(self, X: csr_matrix, user_ids: np.array = None) -> csr_matrix:
        """
        For each user predict the K most popular items.
        :param X: Sparse user-item matrix which will be used to do the predictions; only the set of users will be used.
        :param user_ids: Unused parameter.
        :return: User-item matrix with the prediction scores as values.
        """
        check_is_fitted(self)
        users, items = X.shape

        # TODO: filter already seen items from recommendation
        U = X.nonzero()[0]
        U_conf = self._generate_confidence(X)
        U_user_factors = np.zeros((users, self.factors))
        self._least_squares(U_conf, U_user_factors, self.item_factors_)

        score_list = []
        for u in set(U):
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
          - Binary higher mean: c_ui = round(r_ui / r_max)
          - None: c_ui = r_ui
        :param r: User-item matrix which the calculations are based on.
        :return: User-item matrix converted with the confidence values.
        """
        result = csr_matrix(r, copy=True)
        if self.confidence_scheme == "minimal":
            for i, value in enumerate(result.data):
                result.data[i] = 1 + self.alpha * value
        elif self.confidence_scheme == "log-scaling":
            for i, value in enumerate(result.data):
                result.data[i] = 1 + self.alpha * np.log(1 + value / self.epsilon)
        elif self.confidence_scheme == "binary-higher-mean":
            value_max = max(result.data)
            for i, value in enumerate(result.data):
                result.data[i] = 1 if value > value_max / 2 else 0
        elif self.confidence_scheme == 'none':
            pass
        else:
            raise ValueError("Invalid confidence scheme parameter.")

        return result

    def _alternating_least_squares(self):
        """
        The ALS algorithm will execute the least squares calculation for x number of iterations.
        According factorizing matrix C into two factors Users and Items such that R \approx U^T I.
        """
        users, items = self.c.shape

        self.user_factors_ = np.random.rand(users, self.factors).astype(np.float32) * 0.01
        self.item_factors_ = np.random.rand(items, self.factors).astype(np.float32) * 0.01

        c = self.c.tocsr()
        ct = self.c.T.tocsr()
        for i in tqdm(range(self.iterations)):
            old_uf = np.array(self.user_factors_, copy=True)
            old_if = np.array(self.item_factors_, copy=True)

            self._least_squares(c, self.user_factors_, self.item_factors_)
            self._least_squares(ct, self.item_factors_, self.user_factors_)

            norm_uf = np.linalg.norm(old_uf - self.user_factors_, 2)
            norm_if = np.linalg.norm(old_if - self.item_factors_, 2)
            logger.debug(
                f"{self.name} - Iteration {i} - L2-norm of diff user_factors: {norm_uf} - L2-norm of diff "
                f"item_factors: {norm_if}")

    def _least_squares(self, matrix_c: csr_matrix, X: np.ndarray, Y: np.ndarray):
        """
        Least squares algorithm to calculate the 'new' user/item factors.
        :param matrix_c: Confidence matrix
        :param X: Factor matrix which need to be recalculated
        :param Y: Based on this factor matrix
        """
        users, factors = X.shape
        YtY = Y.T.dot(Y)

        for u in range(users):
            # accumulate YtCuY + regularization * I in A
            A = YtY + self.regularization * np.eye(factors)

            # accumulate YtCuPu in b
            b = np.zeros(factors)

            for i, confidence in nonzeros(matrix_c, u):
                factor = Y[i]
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor

            # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
            X[u] = np.linalg.solve(A, b)
