# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNVaz(TARSItemKNN):
    """Time aware variant of ItemKNN which uses a exponential decay function and pearson similarity.

    Algorithm as described in
    Understanding the Temporal Dynamics of Recommendations across Different Rating Scales
    Paula Cristina Vaz, Ricardo Ribeiro, David Martins de Matos.
    Late-Breaking Results, Project Papers and Workshop Proceedings of the 21st Conference
    on User Modeling, Adaptation, and Personalization. Rome, Italy, June 10-14, 2013.

    The algorithm uses an exponential decay function:

    .. math::

        \\Gamma(x) = e^{- \\alpha \\cdot \\text{x}}

    where :math:`\\alpha` is the decay scaling parameter,
    and x is the time between the maximal timestamp in the matrix
    and the timestamp of the event.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to 1/(24*3600). 
        This means for every day since an interaction, the value of it will be divided by 'e'.
    :type fit_decay: float, optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to 1/(24*3600).
        This means for every day since an interaction, the value of it will be divided by 'e'.
    :type predict_decay: float, optional
    """

    def __init__(self, K: int = 200, fit_decay: float = 1 / (24 * 3600), predict_decay: float = 1 / (24 * 3600)):
        super().__init__(K, fit_decay, predict_decay, similarity="pearson", decay_function="exponential")
