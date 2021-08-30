import numpy as np
import scipy.sparse

from recpack.algorithms.base import (
    ItemSimilarityMatrixAlgorithm,
)
from recpack.algorithms import SVD, NMF


class NMFItemToItem(ItemSimilarityMatrixAlgorithm):
    """Algorithm using similarities between NMF item embeddings.

    Item embeddings are computed using the NMF algorithm,
    then a similarity matrix is constructed by
    computing the dot product between the embeddings.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import NMFItemToItem

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        algo = NMFItemToItem(num_components=2)
        # Fit algorithm
        algo.fit(X)

        # After fitting an item similarity matrix is fitted
        print(algo.similarity_matrix_.shape)
        # (3, 3)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()


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

        self.similarity_matrix_ = (
            self.model_.item_embedding_.T @ self.model_.item_embedding_
        )
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)


class SVDItemToItem(ItemSimilarityMatrixAlgorithm):
    """Use similarity between item embeddings computed by using SVD.

    Item embeddings are computed using the SVD algorithm,
    the similarities are then computed by the dot product of the
    item embeddings.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import SVDItemToItem

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        algo = SVDItemToItem(num_components=2)
        # Fit algorithm
        algo.fit(X)

        # After fitting an item similarity matrix is fitted
        print(algo.similarity_matrix_.shape)
        # (3, 3)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

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

        self.similarity_matrix_ = (
            self.model_.item_embedding_.T @ self.model_.item_embedding_
        )
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)
