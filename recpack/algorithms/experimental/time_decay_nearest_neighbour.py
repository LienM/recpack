import numpy as np
import warnings
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms.base import Algorithm
from recpack.data.matrix import InteractionMatrix


class TimeDecayingNearestNeighbour(Algorithm):
    """Time Decaying Nearest Neighbours model.

    First described in 'Dynamic Item-Based Recommendation Algorithm with Time Decay'
    Chaolun Xia, Xiaohong Jiang, Sen Liu, Zhaobo Luo, Zhang Yu,
    2010 Sixth International Conference on Natural Computation (ICNC 2010)

    For each item the K most similar items are computed during fit.
    Decay function parameter decides how to compute the similarity between two items.

    .. math::
        \\sim(i,j) = \\sum_{k=1}^{n} R_{k,i} . R_{k,j} . \\theta(|T_{k,i} - T_{k,j}|)

    Supported options are: ``"concave"``, ``"convex"`` and ``"linear"``

    - Concave decay function between item i and j is computed as
      the ``decay_coeff^t``. With t the absolute time interval between the interactions on both items.

    **Example of use**::

        import numpy as np
        from recpack.data.matrix import InteractionMatrix
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
        make sure to pick a value in the correct interval for the selected decay function.
        Defaults to 0.5
    :type decay_coeff: float, optional
    :param decay_fn: The decay function that needs to be applied on the item similarity scores.
        Defaults to concave
    :type decay_fn: str, optional
    """

    # TODO - Add time interval step size
    SUPPORTED_COEFF_RANGES = {
        "concave": lambda x: 0 <= x <= 1,
        "convex": lambda x: 0 < x < 1,
        "linear": lambda x: 0 <= x <= 1
    }

    """The supported Decay function options"""

    def __init__(self, K=200, decay_coeff: float = 0.5, decay_fn: str = "concave"):
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

    def _concave_matrix_decay(self, X: csr_matrix, max_delta: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = self.decay_coeff ** X.data
        return X_copy

    def _convex_matrix_decay(self, X: csr_matrix, max_delta: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = (1 - (self.decay_coeff ** (max_delta - X.data)))
        return X_copy

    def _linear_matrix_decay(self, X: csr_matrix, max_delta: int) -> csr_matrix:
        X_copy = X.copy()
        X_copy.data = 1 - (X.data / max_delta) * self.decay_coeff
        return X_copy

    def _compute_dynamic_similarity_with_decay(self, X: InteractionMatrix, max_delta: int):

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
            cooc_decayed = SUPPORTED_DECAY_FUNCTIONS[self.decay_fn](cooc_time_delta, max_delta)
            item_similarities += cooc_decayed

        return item_similarities.tocsr()

    def _fit(self, X: InteractionMatrix):
        # X.timestamps gives a pandas MultiIndex object, indexed by user and item,
        # we drop the index, and group by just the item index
        # then we select the maximal timestamp from this groupby
        largest_time_interval = None
        if self.decay_fn in ("convex", "linear"):
            max_ts = X.timestamps.reset_index()['ts'].max()
            min_ts = X.timestamps.reset_index()['ts'].min()

            largest_time_interval = max_ts - min_ts

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
