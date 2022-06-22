from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.matrix import InteractionMatrix
from recpack.util import get_top_K_values


class TimeDecayingNearestNeighbour(TopKItemSimilarityMatrixAlgorithm):
    """Time Decaying Nearest Neighbours model.

    First described in 'Dynamic Item-Based Recommendation Algorithm with Time Decay'
    Chaolun Xia, Xiaohong Jiang, Sen Liu, Zhaobo Luo, Zhang Yu,
    2010 Sixth International Conference on Natural Computation (ICNC 2010)

    For each item the K most similar items are computed during fit.
    Decay function parameter decides how to compute the similarity between two items.

    .. math::
        \\text{sim}(i,j) = \\sum\\limits_{u=1}^{|U|} R_{u,i} \\cdot R_{u,j} \\cdot \\theta(|T_{u,i} - T_{u,j}|)

    Supported options are: ``"concave"``, ``"convex"`` and ``"linear"``

    - Concave decay function between item i and j is computed as:
    .. math::
        \\theta(x) = \\alpha^{x} \\text{for} \\alpha \\in  [0, 1]

    - Convex decay function between item i and j is computed as:
    .. math::
        \\theta(x) = 1 - \\beta^{t-x} \\text{for} \\beta \\in  (0, 1)

    - Linear decay function between item i and j is computed as:
    .. math::
        \\theta(x) = 1 - \\frac{x}{t} \\cdot \\gamma \\text{for} \\gamma \\in  [0, 1]

    With ``t`` the absolute time interval between the interactions on both items.

    **Example of use**::

        import numpy as np
        from recpack.matrix import InteractionMatrix
        from recpack.algorithms.experimental import TimeDecayingNearestNeighbour

        USER_IX = InteractionMatrix.USER_IX
        ITEM_IX = InteractionMatrix.ITEM_IX
        TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX

        data = {
            TIMESTAMP_IX: [1, 1, 1, 2, 3, 4],
            ITEM_IX: [0, 0, 1, 2, 1, 2],
            USER_IX: [0, 1, 2, 2, 1, 0]
        }
        df = pd.DataFrame.from_dict(data)

        X = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

        # We'll only keep the closest neighbour for each item.
        # Default uses concave decay function with coefficient equal to 0.5
        algo = TimeDecayingNearestNeighbour(K=1)
        # Fit algorithm
        algo.fit(X)

        # We can inspect the fitted model
        print(algo.similarity_matrix_.nnz)
        # 3

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param decay_coeff: How strongly the decay function should influence the scores,
        make sure to pick a value in the correct interval
        for the selected decay function.
        Defaults to 0.5
    :type decay_coeff: float, optional
    :param decay_fn: The decay function that needs to
        be applied on the item similarity scores.
        Defaults to concave
    :type decay_fn: str, optional
    :param decay_interval: Defines the basic time interval unit in seconds.
        Defaults to 24*3600.
    :typ decay_interval: Optional[int]
    """

    SUPPORTED_COEFF_RANGES = {
        "concave": lambda x: 0 <= x <= 1,
        "convex": lambda x: 0 < x < 1,
        "linear": lambda x: 0 <= x <= 1,
    }

    """The supported Decay function options"""

    def __init__(
        self,
        K: int = 200,
        decay_coeff: float = 0.5,
        decay_fn: str = "concave",
        decay_interval: int = 24 * 3600,
    ):
        super().__init__(K=K)

        if decay_fn not in self.SUPPORTED_COEFF_RANGES.keys():
            raise ValueError(f"decay_function {decay_fn} not supported")
        self.decay_fn = decay_fn

        if not self.SUPPORTED_COEFF_RANGES[decay_fn](decay_coeff):
            raise ValueError(f"decay_coeff {decay_coeff} is not in the supported range")
        self.decay_coeff = decay_coeff

        if decay_interval <= 0 or type(decay_interval) == float:
            raise ValueError("Decay_interval needs to be a positive nonzero integer")
        self.decay_interval = decay_interval

    def _transform_fit_input(self, X):
        # X needs to be an InteractionMatrix for us to have access to
        # the time of interaction at fitting time
        assert isinstance(X, InteractionMatrix)
        # X needs to have timestamps available
        assert X.has_timestamps
        # No transformation needed
        return X

    def _concave_matrix_decay(self, X: csr_matrix, max_delta: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = self.decay_coeff ** (X.data / self.decay_interval)
        return X_copy

    def _convex_matrix_decay(self, X: csr_matrix, max_delta: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = 1 - (self.decay_coeff ** ((max_delta - X.data) / self.decay_interval))
        return X_copy

    def _linear_matrix_decay(self, X: csr_matrix, max_delta: int) -> csr_matrix:
        X_copy = X.copy()
        # The interval does not impact the linear decay,
        # since max_delta and x are assumed to be in the same interval.
        # So the factor in nominator and denominator cancel out.
        X_copy.data = 1 - (X.data / max_delta) * self.decay_coeff
        return X_copy

    def _compute_dynamic_similarity_with_decay(self, X: InteractionMatrix, max_delta: int):

        SUPPORTED_DECAY_FUNCTIONS = {
            "concave": self._concave_matrix_decay,
            "convex": self._convex_matrix_decay,
            "linear": self._linear_matrix_decay,
        }

        num_users, num_items = X.shape

        # Get the timestamps multi index
        last_timestamps_matrix = X.last_timestamps_matrix

        item_similarities = csr_matrix((num_items, num_items))
        for user in tqdm(X.active_users):
            # Construct user history as np array
            user_hist = last_timestamps_matrix[user, :].T

            # Compute the Cooc matrix for this user,
            # with the difference in timestamp as value.
            # 1. compute cooc matrix,
            #   such that cooc_one_ts[i,j] = t(j) if hist[i] and hist[j]
            cooc_one_ts = user_hist.astype(bool) @ (user_hist.T)

            # 2. construct the cooc matrix with timsteamps of item i
            cooc_other_ts = cooc_one_ts.astype(bool).multiply(user_hist)

            # 3. Construct cooc csr matrix with the time delta between interactions
            cooc_time_delta = csr_matrix(
                abs(cooc_one_ts - cooc_other_ts),
            )

            # 4. apply the decay on these values
            cooc_decayed = SUPPORTED_DECAY_FUNCTIONS[self.decay_fn](cooc_time_delta, max_delta)
            item_similarities += cooc_decayed

        return item_similarities

    def _fit(self, X: InteractionMatrix):
        # X.timestamps gives a pandas MultiIndex object, indexed by user and item,
        # we drop the index, and group by just the item index
        # then we select the maximal timestamp from this groupby
        largest_time_interval = None
        if self.decay_fn in ("convex", "linear"):
            max_ts = X.timestamps.max()
            min_ts = X.timestamps.min()

            largest_time_interval = max_ts - min_ts

        item_similarities = self._compute_dynamic_similarity_with_decay(X, largest_time_interval)

        self.similarity_matrix_ = get_top_K_values(item_similarities, self.K)
