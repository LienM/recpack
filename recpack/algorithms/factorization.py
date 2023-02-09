# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from typing import Optional
from scipy.sparse import csr_matrix, diags
import sklearn.decomposition

from recpack.algorithms.base import (
    FactorizationAlgorithm,
)


class NMF(FactorizationAlgorithm):
    """Non negative matrix factorization.

    Matrix is decomposed into a user embedding and an item embedding of the same size.
    Decomposition happens using the NMF implementation from sklearn.

    Uses the NMF implementation from
    `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html>`_.

    :param num_components: The size of the latent dimension
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison,
                            defaults to None
    :type seed: int, optional
    :param alpha: Regularization parameter, defines how much regularization is applied.
    :type alpha: float, optional
    :param l1_ratio: Defines how much L1 normalisation is used,
        compared to L2 normalisation.
        Value between 1 and 0, where 1 means only L1 normalisation,
        and 0 only L2 normalisation.
        Defaults to 0
    :type l1_ratio: float, optional
    """

    def __init__(
        self, num_components: int = 100, seed: Optional[int] = None, alpha: float = 0.0, l1_ratio: float = 0.0
    ):
        super().__init__(num_components)
        self.seed = seed
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _fit(self, X: csr_matrix):

        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(
            n_components=self.num_components,
            init="random",
            random_state=self.seed,
            alpha_H=self.alpha,
            alpha_W=self.alpha,
            l1_ratio=self.l1_ratio,
        )

        # Factorization is W * H. Where W contains user latent vectors, and H
        # contains item latent vectors
        self.user_embedding_ = model.fit_transform(X)
        self.item_embedding_ = model.components_

        # Post conditions
        assert self.user_embedding_.shape == (X.shape[0], self.num_components)
        assert self.item_embedding_.shape == (self.num_components, X.shape[1])


class SVD(FactorizationAlgorithm):
    """Singular Value Decomposition used as a matrix factorization algorithm.

    The SVD is computed using
    `TruncatedSVD <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_
    from sklearn.
    This computes three matrices :math:`U, \\Sigma, V` such that
    :math:`U \\times \\Sigma \\times V = X`.

    Matrix :math:`U` is used as the user embedding and the item embeddings are computed as :math:`\\Sigma \\times V`.

    :param num_components: The size of the embeddings.
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison, defaults to None
    :type seed: int, optional
    """

    def __init__(self, num_components: int = 100, seed: Optional[int] = None):
        super().__init__(num_components=num_components)

        self.seed = seed

    def _fit(self, X: csr_matrix):

        model = sklearn.decomposition.TruncatedSVD(n_components=self.num_components, n_iter=7, random_state=self.seed)
        # Factorization computes U x Sigma x V
        # U are the user features,
        # Sigma x V are the item features.
        self.user_embedding_ = model.fit_transform(X)

        V = model.components_
        sigma = diags(model.singular_values_)
        self.item_embedding_ = sigma @ V

        # Post conditions
        assert self.user_embedding_.shape == (X.shape[0], self.num_components)
        assert self.item_embedding_.shape == (self.num_components, X.shape[1])
