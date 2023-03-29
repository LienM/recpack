# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNVaz(TARSItemKNN):
    """Time decayed similarity algorithm with pearson similarity.

    Algorithm as described in
    Understanding the Temporal Dynamics of Recommendations across Different Rating Scales
    Paula Cristina Vaz, Ricardo Ribeiro, David Martins de Matos.
    Late-Breaking Results, Project Papers and Workshop Proceedings of the 21st Conference
    on User Modeling, Adaptation, and Personalization. Rome, Italy, June 10-14, 2013.

    :param K: The number of neighbours to compute per item. Defaults to 200.
    :type K: int, optional
    # TODO Describe what this decay means?
    :param fit_decay: Decay parameter used during fitting.
        Defaults to 1/(24*3600).
    :type fit_decay: float, optional
    :param predict_decay: Decay parameter used during predicting.
        Defaults to 1/(24*3600).
    :type predict_decay: float, optional
    """

    def __init__(self, K: int = 200, fit_decay: float = 1 / (24 * 3600), predict_decay: float = 1 / (24 * 3600)):
        super().__init__(K, fit_decay, predict_decay, similarity="pearson", decay_function="exponential")
