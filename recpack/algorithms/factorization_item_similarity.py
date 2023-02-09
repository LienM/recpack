# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import scipy.sparse

from recpack.algorithms.base import (
    ItemSimilarityMatrixAlgorithm,
)
from recpack.algorithms import SVD, NMF


class NMFItemToItem(ItemSimilarityMatrixAlgorithm):
    """Computes similarities between items as the similarity between their NMF item embeddings.

    First, item embeddings are computed using the :class:`recpack.algorithms.NMF` algorithm.
    The similarity matrix is constructed by computing the dot product between the item embeddings.

    :param num_components: The size of the latent dimension
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison,
                            defaults to None
    :type seed: int, optional
    """

    def __init__(self, num_components: int = 100, seed: int = None):
        super().__init__()
        self.num_components = num_components
        self.seed = seed

    def _fit(self, X: scipy.sparse.csr_matrix):
        self.model_ = NMF(self.num_components, self.seed)
        self.model_.fit(X)

        self.similarity_matrix_ = self.model_.item_embedding_.T @ self.model_.item_embedding_
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)


class SVDItemToItem(ItemSimilarityMatrixAlgorithm):
    """Computes similarities between items as the similarity between their SVD embeddings.

    First, item embeddings are computed using the :class:`recpack.algorithms.SVD` algorithm.
    The similarity matrix is constructed by computing the dot product between the item embeddings.

    :param num_components: The size of the latent dimension
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison,
                            defaults to None
    :type seed: int, optional
    """

    def __init__(self, num_components: int = 100, seed: int = None):
        super().__init__()
        self.num_components = num_components
        self.seed = seed

    def _fit(self, X: scipy.sparse.csr_matrix):
        self.model_ = SVD(self.num_components, self.seed)
        self.model_.fit(X)

        self.similarity_matrix_ = self.model_.item_embedding_.T @ self.model_.item_embedding_
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)
