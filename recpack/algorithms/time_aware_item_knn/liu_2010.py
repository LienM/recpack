# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Module with time-dependent ItemKNN implementations"""

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNLiu(TARSItemKNN):
    """Time aware variant of ItemKNN which uses an exponential decay function and cosine similarity.

    Algorithm as described in
    Nathan N. Liu, Min Zhao, Evan Xiang, and Qiang Yang.
    2010. Online evolutionary collaborative filtering.
    In Proceedings of the fourth ACM conference on Recommender systems (RecSys '10).
    Association for Computing Machinery, New York, NY, USA, 95–102.
    https://doi.org/10.1145/1864708.1864729

    The algorithm uses an exponential decay function:

    .. math::

        \\Gamma(x) = e^{- \\alpha \\cdot \\text{x}}

    where :math:`\\alpha` is the decay scaling parameter,
    and x is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    Similarity is computed on this weighted matrix, using cosine similarity.
    At prediction time a user's history is weighted using the same formula with a different alpha.
    This weighted history is then multiplied with the precomputed similarity matrix.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to 1 / (24 * 3600).
        This means for every day since an interaction, the value of it will be divided by 'e'.
    :type fit_decay: float, optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to 1 / (24 * 3600).
        This means for every day since an interaction, the value of it will be divided by 'e'.
    :type predict_decay: float, optional
    """

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / (24 * 3600),
        predict_decay: float = 1 / (24 * 3600),
    ):
        super().__init__(
            K=K, fit_decay=fit_decay, predict_decay=predict_decay, similarity="cosine", decay_function="exponential"
        )
