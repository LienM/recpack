import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.time_aware_item_knn.base import TARSItemKNNCoocDistance
from recpack.algorithms.util import invert
from recpack.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values

# TODO: unify this one with Xia?

EPSILON = 1e-13


class TARSItemKNNHermann(TARSItemKNNCoocDistance):
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

    :param K: number of neighbours to store while training. Defaults to 200.
    :type K: int, optional
    :param decay_interval: The amount of seconds to consider as a single unit of time.
        Defaults to 1
    :type decay_interval: int, optional
    """

    def __init__(self, K: int = 200, decay_interval: int = 1):
        super().__init__(
            K,
            fit_decay=1,
            predict_decay=0,
            decay_interval=decay_interval,
            similarity="hermann",
            decay_function="inverse",
            event_age_weight=1,
        )
