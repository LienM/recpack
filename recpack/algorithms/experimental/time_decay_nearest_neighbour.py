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

    def _concave_matrix_decay(self, X: csr_matrix, t: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = self.decay_coeff ** X.data
        return X_copy

    def _convex_matrix_decay(self, X: csr_matrix, t: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = (1 - (self.decay_coeff ** (t - X.data)))
        return X_copy

    def _linear_matrix_decay(self, X: csr_matrix, t: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = 1 - (X.data / t) * self.decay_coeff
        return X_copy

    def _compute_dynamic_similarity_with_decay(self, X: InteractionMatrix, t: int):

        SUPPORTED_DECAY_FUNCTIONS = {
            "concave": self._concave_matrix_decay,
            "convex": self._convex_matrix_decay,
            "linear": self._linear_matrix_decay
        }

        num_users, num_items = X.shape

        # Get the timestamps multi index
        timestamps = X.timestamps

        item_similarities = lil_matrix((num_items, num_items))
        for user in X.active_users:
            # Construct user history as np array
            user_hist = np.zeros((X.shape[1], 1))
            user_hist[timestamps[user].index.values, 0] = X.timestamps[user].values

            # Compute the Cooc matrix for this user, with the difference in timestamp as value.
            # 1. compute cooc matrix, such that cooc_one_ts[i,j] = t(j) if hist[i] and hist[j]
            cooc_one_ts = user_hist.astype(bool) @ user_hist.T
            # 2. Construct cooc csr matrix with the time delta between interactions
            cooc_time_delta = csr_matrix(abs((cooc_one_ts - user_hist) * cooc_one_ts.astype(bool)))

            # 3. apply the decay on these values
            cooc_decayed = SUPPORTED_DECAY_FUNCTIONS[self.decay_fn](cooc_time_delta, t)
            item_similarities += cooc_decayed

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
