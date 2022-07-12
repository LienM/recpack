import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.util import invert
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values

# TODO: unify this one with Xia?

EPSILON = 1e-13


class TARSItemKNNHermann(TopKItemSimilarityMatrixAlgorithm):
    """ItemKNN algorithm with temporal decay,
    taking into account age of events and pairwise distance between cooccurences.

    Presented in Hermann, Christoph. "Time-based recommendations for lecture materials."
    EdMedia+ Innovate Learning. Association for the Advancement of Computing in Education (AACE), 2010.

    Similarity between two items is computed as the avg of :math:`S_{u,i,j}`
    for each user that has seen both items i and j.

    .. math::

        s_{u,i,j} = \\frac{1}{\\Delta t_{u,i,j} + \\Delta d_{u,i,j}}

    where :math:`\\Delta t_{u,i,j}` is the distance in time units between the user interacting with item i and j.
    :math:`\\Delta d_{u,i,j}` is the maximal distance in time units between a user interactions with i or j to now.

    :param K: number of neighbours to store while training.
    :type K: int
    :param time_unit_seconds: The amount of seconds to consider as a single unit of time.
        Defaults to 1
    :type time_unit_seconds: int, optional
    """

    def __init__(self, K: int, time_unit_seconds: int = 1):
        super().__init__(K)
        self.time_unit_seconds = time_unit_seconds

    def _transform_fit_input(self, X: Matrix):
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: InteractionMatrix):
        num_users, num_items = X.shape

        # Get the timestamps multi index
        last_timestamps_matrix = X.last_timestamps_matrix
        now = (last_timestamps_matrix.max() + 1) / self.time_unit_seconds  # Doing +1s to avoid disappearing events

        # Rescale the timestamps matrix to be in the right 'time units'
        last_timestamps_matrix = last_timestamps_matrix / self.time_unit_seconds
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
            # By adding a small value to one of the timestamps, we avoid vanishing zero distances.
            cooc_other_ts.data = cooc_other_ts.data + EPSILON

            # 3. Construct cooc csr matrix with the time delta between interactions
            cooc_time_delta = csr_matrix(
                abs(cooc_one_ts - cooc_other_ts),
            )

            # 4. Compute the maximal timedelta with t_0
            cooc_distance_to_now = (cooc_one_ts < cooc_other_ts).multiply(cooc_one_ts) + (
                cooc_other_ts < cooc_one_ts
            ).multiply(cooc_other_ts)
            cooc_distance_to_now.data = now - cooc_distance_to_now.data

            # Compute similarity contribution as 1/(delta_t + delta_d)
            similarity_contribution = invert(cooc_time_delta + cooc_distance_to_now)
            item_similarities += similarity_contribution

        # normalise the similarities as per the paper, using number of users that had a cooc.
        cooc = csr_matrix(X.binary_values.T @ X.binary_values)
        item_similarities = item_similarities.multiply(invert(cooc))
        item_similarities[np.arange(num_items), np.arange(num_items)] = 0

        self.similarity_matrix_ = get_top_K_values(csr_matrix(item_similarities), self.K)


class TARSItemKNNHermannExtension(TopKItemSimilarityMatrixAlgorithm):
    """ItemKNN algorithm with temporal decay,
    taking into account age of events and pairwise distance between cooccurences.

    Inspired by Hermann, Christoph. "Time-based recommendations for lecture materials."
    EdMedia+ Innovate Learning. Association for the Advancement of Computing in Education (AACE), 2010.

    Instead of the presented similarity, the similarity is computed using a
    conditional probability inspired similarity:

    .. math::

        s_{i,j} = \\frac{1}{|X_i|}\\sum\\limits_{u \\in \\mathcal{U}} \\frac{X_{u,i} X_{u,j}}{\\Delta t_{u,i,j} + \\Delta d_{u,i,j}}

    where :math:`\\Delta t_{u,i,j}` is the distance in time units between the user interacting with item i and j.
    :math:`\\Delta d_{u,i,j}` is the maximal distance in time units between a user interactions with i or j to now.

    :param K: number of neighbours to store while training.
    :type K: int
    :param time_unit_seconds: The amount of seconds to consider as a single unit of time.
        Defaults to 1
    :type time_unit_seconds: int, optional
    """

    def __init__(self, K: int, time_unit_seconds: int = 1):
        super().__init__(K)
        self.time_unit_seconds = time_unit_seconds

    def _transform_fit_input(self, X: Matrix):
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: InteractionMatrix):
        num_users, num_items = X.shape

        # Get the timestamps multi index
        last_timestamps_matrix = X.last_timestamps_matrix
        now = (last_timestamps_matrix.max() + 1) / self.time_unit_seconds  # Doing +1s to avoid disappearing events

        # Rescale the timestamps matrix to be in the right 'time units' TODO: add a test
        last_timestamps_matrix = last_timestamps_matrix / self.time_unit_seconds
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
            # By adding a small value to one of the timestamps, we avoid vanishing zero distances.
            cooc_other_ts.data = cooc_other_ts.data + EPSILON

            # 3. Construct cooc csr matrix with the time delta between interactions
            cooc_time_delta = csr_matrix(
                abs(cooc_one_ts - cooc_other_ts),
            )

            # 4. Compute the maximal timedelta with t_0
            cooc_distance_to_now = (cooc_one_ts < cooc_other_ts).multiply(cooc_one_ts) + (
                cooc_other_ts < cooc_one_ts
            ).multiply(cooc_other_ts)
            cooc_distance_to_now.data = now - cooc_distance_to_now.data

            # Compute similarity contribution as 1/(delta_t + delta_d)
            similarity_contribution = invert(cooc_time_delta + cooc_distance_to_now)
            item_similarities += similarity_contribution

        # normalise the similarities using the amount of visits of i.
        item_similarities = item_similarities.multiply(invert(X.binary_values.sum(axis=0)))
        item_similarities[np.arange(num_items), np.arange(num_items)] = 0

        self.similarity_matrix_ = get_top_K_values(csr_matrix(item_similarities), self.K)
