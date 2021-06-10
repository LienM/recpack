import numpy as np
import warnings
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms.base import Algorithm
from recpack.data.matrix import InteractionMatrix

from itertools import permutations


class TimeDecayingNearestNeighbour(Algorithm):

    # TODO - Add time interval step size
    SUPPORTED_COEFF_RANGES = {
        "concave": lambda x: 0 <= x <= 1,
        "convex": lambda x: 0 < x < 1,
        "linear": lambda x: 0 <= x <= 1
    }

    """The supported Decay function options"""

    def __init__(self, K=200, decay_coeff=1, decay_fn: str = "concave"):
        self.K = K

        if decay_fn not in self.SUPPORTED_COEFF_RANGES.keys():
            raise ValueError(f"decay_function {decay_fn} not supported")
        self.decay_fn = decay_fn

        if not self.SUPPORTED_COEFF_RANGES[decay_fn](decay_coeff):
            raise ValueError(f"decay_coeff {decay_coeff} is not in the supported range")
        self.decay_coeff = decay_coeff

    def _transform_fit_input(self, X):
        # X needs to be an InteractionMatrix for us to have access to
        # the time of interaction at fitting time
        assert isinstance(X, InteractionMatrix)
        # X needs to have timestamps available
        assert X.has_timestamps
        # No transformation needed
        return X

    def _concave_time_decay(self, x: int, t: int) -> float:
        return self.decay_coeff ** x

    def _convex_time_decay(self, x: int, t: int) -> float:
        return (1 - (self.decay_coeff ** (t - x)))

    def _linear_time_decay(self, x: int, t: int) -> float:
        if t == 0:
            return 0
        return 1 - (x / t) * self.decay_coeff

    def _compute_dynamic_similarity_with_decay(self, X: InteractionMatrix, t: int):

        SUPPORTED_DECAY_FUNCTIONS = {
            "concave": self._concave_time_decay,
            "convex": self._convex_time_decay,
            "linear": self._linear_time_decay
        }

        num_users, num_items = X.shape
        csr = X.binary_values
        item_similarities = lil_matrix((num_items, num_items), dtype=np.float64)

        df = X._df

        for i, j in permutations(range(0, num_items), 2):
            for k in range(0, num_users):
                if csr[k, i] and csr[k, j]:
                    cond1 = (df.iid == i) & (df.uid == k)
                    cond2 = (df.iid == j) & (df.uid == k)

                    item_similarities[i, j] += csr[k, i] * csr[k, j] * SUPPORTED_DECAY_FUNCTIONS[self.decay_fn](abs(df[cond1].iloc[0]['ts'] - df[cond2].iloc[0]['ts']), t)

        return item_similarities.tocsr()

    def _fit(self, X: InteractionMatrix):
        # X.timestamps gives a pandas MultiIndex object, indexed by user and item,
        # we drop the index, and group by just the item index
        # then we select the maximal timestamp from this groupby
        largest_time_interval = None
        if self.decay_fn in ("convex", "linear"):
            max_ts_per_item = X.timestamps.reset_index()['ts'].max()
            min_ts_per_item = X.timestamps.reset_index()['ts'].min()

            largest_time_interval = max_ts_per_item - min_ts_per_item

        item_similarities = self._compute_dynamic_similarity_with_decay(X, largest_time_interval)

        self.similarity_matrix_ = item_similarities

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_

        # If self.similarity_matrix_ is not a csr matrix,
        # scores will also not be a csr matrix
        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar items for each item

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar items for {missing} items.")
