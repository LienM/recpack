import logging

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from recpack.algorithms import Algorithm

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
    and as such leaving the HP to the defaults works good enough.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import WMF

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        algo = WMF()
        # Fit algorithm
        algo.fit(X)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param conficence_scheme: Which confidence scheme should be used
        to calculate the confidence matrix.
        Options are ["minimal", "log-scaling"].
        Defaults to "minimal"
    :type conficence_scheme: string, optional
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

    def _fit(self, X: csr_matrix) -> Algorithm:
        """Calculate the user- and item-factors which will approximate X
            after applying a dot-product.

        :param X: Sparse user-item matrix which will be used to fit the algorithm.
        :return: The fitted WeightedMatrixFactorizationAlgorithm itself.
        """
        self.num_users, self.num_items = X.shape
        self.known_users = set(X.nonzero()[0])
        self.user_factors_, self.item_factors_ = self._alternating_least_squares(X)

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Prediction scores are calculated as the dotproduct of
            the recomputed user-factors and the item-factors.

        :param X: Sparse user-item matrix which will be used to do the predictions;
            only for set of users with interactions will recommendations be generated.
        :return: User-item matrix with the prediction scores as values.
        """

        U = set(X.nonzero()[0])
        U_conf = self._generate_confidence(X)
        U_user_factors = self._least_squares(
            U_conf, self.item_factors_, self.num_users, U
        )

        score_matrix = csr_matrix(U_user_factors @ self.item_factors_.T)

        self._check_prediction(score_matrix, X)
        return score_matrix

    def _generate_confidence(self, r) -> csr_matrix:
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
        :return: User-item matrix converted with the confidence values.
        """
        result = csr_matrix(r, copy=True)
        if self.confidence_scheme == "minimal":
            result.data = self.alpha * result.data
        elif self.confidence_scheme == "log-scaling":
            result.data = self.alpha * np.log(1 + result.data / self.epsilon)

        return result

    def _alternating_least_squares(self, X: csr_matrix) -> (np.ndarray, np.ndarray):
        """
        The ALS algorithm will execute the least squares calculation for x number of iterations.
        According factorizing matrix C into two factors Users and Items such that R \approx U^T I.
        :param X: Sparse matrix which the ALS algorithm should be applied on.
        :return: Generated user- and item-factors based on the input matrix X.
        """
        user_factors = (
            np.random.rand(self.num_users, self.num_components).astype(np.float32)
            * 0.01
        )
        item_factors = (
            np.random.rand(self.num_items, self.num_components).astype(np.float32)
            * 0.01
        )

        c = self._generate_confidence(X)
        ct = c.T.tocsr()
        item_set = set(range(self.num_items))

        for i in tqdm(range(self.iterations)):
            old_uf = np.array(user_factors, copy=True)
            old_if = np.array(item_factors, copy=True)

            user_factors = self._least_squares(
                c, item_factors, self.num_users, self.known_users
            )
            item_factors = self._least_squares(
                ct, user_factors, self.num_items, item_set
            )

            norm_uf = np.linalg.norm(old_uf - user_factors, "fro")
            norm_if = np.linalg.norm(old_if - item_factors, "fro")
            logger.debug(
                f"{self.name} - Iteration {i} - L2-norm of diff user_factors: {norm_uf} - L2-norm of diff "
                f"item_factors: {norm_if}"
            )

        return user_factors, item_factors

    def _least_squares(
        self,
        conf_matrix: csr_matrix,
        factors: np.ndarray,
        dimension: int,
        distinct_set: set,
    ) -> np.ndarray:
        """
        Calculate the other factor based on the confidence matrix and the factors with the least squares algorithm.
        It is a general function for item- and user-factors. Depending on the parameter factor_type the other factor
        will be calculated.
        :param conf_matrix: (Transposed) Confidence matrix
        :param factors: Factor array
        :param dimension: User/item dimension.
        :param distinct_set: Set of users/items
        :return: Other factor nd-array based on the factor array and the confidence matrix
        """
        factors_x = np.zeros((dimension, self.num_components))
        YtY = factors.T @ factors

        for i in distinct_set:
            factors_x[i] = self._linear_equation(factors, YtY, conf_matrix, i)

        return factors_x

    def _linear_equation(
        self, Y: np.ndarray, YtY: np.ndarray, C: csr_matrix, x: int
    ) -> np.ndarray:
        """
        Helper function to compute the linear equation used in the Least Squares calculations.
        :param Y: Input factor array
        :param YtY: Product of Y transpose and Y.
        :param C: The (transposed) confidence matrix
        :param x: Calculation for which item/user x
        :return: Solution for the linear equation (YtCxY + regularization * I)^-1 (YtCxPx)
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
        A = YtY + (Y.T @ Cx) @ Y + self.regularization * np.eye(self.num_components)

        # accumulate Yt(Cx + I)Px in b
        cx[cx > 0] = 1  # now Px is represented as cx
        b = (Y.T @ (Cx + eye(Cx.shape[0]))) @ cx.T.toarray()

        # Xu = (YtCxY + regularization * I)^-1 (YtCxPx)
        #  Flatten the result to make sure the dimension is (self.num_components,)
        return np.linalg.solve(A, b).flatten()
