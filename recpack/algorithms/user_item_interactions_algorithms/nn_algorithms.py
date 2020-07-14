import numpy as np
import scipy
from scipy.sparse import diags
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from recpack.algorithms.user_item_interactions_algorithms import (
    SimilarityMatrixAlgorithm,
)


class ItemKNN(SimilarityMatrixAlgorithm):
    def __init__(self, K=200, normalize=False):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
        super().__init__()
        self.K = K
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit a cosine similarity matrix from item to item"""
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        # X.T otherwise we are doing a user KNN
        self.item_cosine_similarities_ = cosine_similarity(X.T, dense_output=False)

        self.item_cosine_similarities_.setdiag(0)
        # Set diagonal to 0, because we don't want to support self similarity

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(
                np.argpartition(self.item_cosine_similarities_.toarray(), -self.K)
            )
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(
            ([1 for i in range(len(indices))], (list(zip(*indices))))
        )
        self.item_cosine_similarities_ = self.item_cosine_similarities_.multiply(mask)

        if self.normalize:
            # normalize per row
            row_sums = self.item_cosine_similarities_.sum(axis=1)
            self.item_cosine_similarities_ = self.item_cosine_similarities_ / row_sums
            self.item_cosine_similarities_ = scipy.sparse.csr_matrix(
                self.item_cosine_similarities_
            )
            self.item_cosine_similarities_.eliminate_zeros()

        return self

    @property
    def sim_matrix(self):
        return self.item_cosine_similarities_


class CosineSimilarity(SimilarityMatrixAlgorithm):
    """ Item cosine similarity based on code of Olivier """

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def fit(self, X, y=None):
        # Base similarity matrix (all dot products)
        similarity = (X.T @ X).toarray()

        # Squared magnitude of preference vectors (number of occurrences)
        square_mag = np.diag(similarity)

        # Inverse squared magnitude
        inv_square_mag = 1 / square_mag
        # If it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        np.fill_diagonal(cosine, 0)

        if self.normalize:
            # normalize per row
            row_sums = cosine.sum(axis=1)
            cosine = cosine / row_sums[:, np.newaxis]

        self.item_cosine_similarities_ = cosine
        return self

    @property
    def sim_matrix(self):
        return self.item_cosine_similarities_


class NotItemKNN(SimilarityMatrixAlgorithm):
    """
    TODO: Figure out what this code is actually implementing. It is not cosine similarity
    It does seem to work fine though.
    """

    def __init__(self, K=200):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
        super().__init__()
        self.K = K

    def fit(self, X):
        """Fit a cosine similarity matrix from item to item"""
        co_mat = X.T @ X
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        A = diags(1 / co_mat.diagonal())

        # This has all item-cosine similarities. Now we should probably set N-K to zero
        self.item_cosine_similarities_ = A @ co_mat

        # Set diagonal to 0, because we don't support self similarity
        self.item_cosine_similarities_.setdiag(0)

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(
                np.argpartition(self.item_cosine_similarities_.toarray(), -self.K)
            )
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(
            ([1 for i in range(len(indices))], (list(zip(*indices))))
        )
        self.item_cosine_similarities_ = self.item_cosine_similarities_.multiply(mask)
        return self

    @property
    def sim_matrix(self):
        return self.item_cosine_similarities_
