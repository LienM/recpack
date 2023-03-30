# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNDing(TARSItemKNN):
    """Time aware variant of ItemKNN which uses an exponential decay function at prediction time and cosine similarity.

    Algorithm as presented in
    Yi Ding and Xue Li. 2005.
    Time weight collaborative filtering.
    In Proceedings of the 14th ACM international conference on Information and knowledge management (CIKM '05).
    Association for Computing Machinery, New York, NY, USA, 485–492.
    https://doi.org/10.1145/1099554.1099689


    Computation of the similarity matrix is the same as normal ItemKNN.
    When predicting however the user's older interactions are given less weight in the final prediction score.

    .. math::

        \\text{sim}(u, i) = \\sum\\limits_{j \\in X_u} e^{-\\alpha \\cdot \\delta t_{u,j}} \\cdot \\text{sim}(i, j)

    Where :math:`\\alpha` is the `predict_decay` parameter.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, Optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to 1 / (24 * 3600).
        This means for every day since an interaction, the value of it will be divided by 'e'.
    :type predict_decay: float, optional
    :param similarity: Which similarity measure to use. Defaults to `"cosine"`.
        ``["cosine", "conditional_probability"]`` are supported.
    :type similarity: str, optional
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]

    def __init__(self, K: int = 200, predict_decay: float = 1 / (24 * 3600), similarity: str = "cosine"):
        super().__init__(K=K, fit_decay=0, predict_decay=predict_decay, similarity=similarity, decay_function="exponential")
