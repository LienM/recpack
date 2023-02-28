# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Module with time-dependent ItemKNN implementations"""

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNLiu(TARSItemKNN):
    """ItemKNN algorithm where older interactions have less weight during both prediction and training.

    Algorithm as defined in Liu, Nathan N., et al. "Online evolutionary collaborative filtering."
    Proceedings of the fourth ACM conference on Recommender systems. 2010.

    Each interaction is decay to

    .. math::

        e^{- \\alpha \\cdot \\text{age}}

    Where alpha is the decay scaling parameter,
    and age is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    Similarity is computed on this weighted matrix, using cosine similarity.

    At prediction time a user's history is weighted using the same formula with a different alpha.
    This weighted history is then multiplied with the precomputed similarity matrix.

    :param K: Amount of neighbours to keep. Defaults to 200.
    :type K: int, Optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to (1/3600).
    :type fit_decay: float, Optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to (1/3600).
    :type predict_decay: float, Optional
    """

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / 3600,
        predict_decay: float = 1 / 3600,
    ):
        super().__init__(K=K, fit_decay=fit_decay, predict_decay=predict_decay, similarity="cosine")
