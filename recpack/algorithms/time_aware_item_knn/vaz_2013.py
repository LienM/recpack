from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNVaz(TARSItemKNN):
    """Time decayed similarity algorithm with pearson similarity.

    Algorithm as described in Vaz, Paula Cristina, Ricardo Ribeiro, and David Martins De Matos.
    "Understanding the Temporal Dynamics of Recommendations across Different Rating Scales."
    UMAP Workshops. 2013.

    :param K: The number of neighbours to compute per item. Defaults to 200.
    :type K: int, optional
    :param fit_decay: decay parameter used during fitting.
        Defaults to 1/(24*3600).
    :type fit_decay: float, optional
    :param predict_decay: decay parameter used during predicting.
        Defaults to 1/(24*3600).
    :type predict_decay: float, optional
    """

    def __init__(self, K: int = 200, fit_decay: float = 1 / (24 * 3600), predict_decay: float = 1 / (24 * 3600)):
        super().__init__(K, fit_decay, predict_decay, similarity="pearson", decay_function="exponential")
