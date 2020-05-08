from collections import Counter, defaultdict
import math
import numpy as np
import scipy
from scipy.sparse import diags
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.user_item_interactions_algorithms import (
    UserItemInteractionsAlgorithm,
)


class ItemKNN(UserItemInteractionsAlgorithm):

    def __init__(self, K=200):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
        self.K = K

    def fit(self, X):
        """Fit a cosine similarity matrix from item to item"""
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        # X.T otherwise we are doing a user KNN
        self.item_cosine_similarities_ = cosine_similarity(X.T, dense_output=False)

        # Set diagonal to 0, because we don't want to support self similarity
        self.item_cosine_similarities_.setdiag(0)

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(np.argpartition(self.item_cosine_similarities_.toarray(), -self.K))
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))))
        self.item_cosine_similarities_ = self.item_cosine_similarities_.multiply(mask)
        return self

    def predict(self, X: scipy.sparse.csr_matrix):
        # Use total sum of similarities
        check_is_fitted(self)
        # TODO: Use average?
        scores = X @ self.item_cosine_similarities_

        if not isinstance(scores, scipy.sparse.csr_matrix):
            scores = scipy.sparse.csr_matrix(scores)

        return scores

    # @property
    # def name(self):
    #     return f"item_knn_{self.K}"


class SharedAccount(ItemKNN):

    def __init__(self, K):
        super().__init__(K)

    def predict(self, X):
        raise NotImplementedError("Under construction, the gnomes are working on it.")


class NotItemKNN(UserItemInteractionsAlgorithm):
    """
    TODO: Figure out what this code is actually implementing. It is not cosine similarity
    It does seem to work fine though.
    """

    def __init__(self, K=200):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
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
            for i, best_items_row in enumerate(np.argpartition(self.item_cosine_similarities_.toarray(), -self.K))
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))))
        self.item_cosine_similarities_ = self.item_cosine_similarities_.multiply(mask)
        return self

    def predict(self, X):
        # Use total sum of similarities
        # TODO: Use average?
        check_is_fitted(self)
        scores = X @ self.item_cosine_similarities_

        if not isinstance(scores, scipy.sparse.csr_matrix):
            scores = scipy.sparse.csr_matrix(scores)

        return scores

    # @property
    # def name(self):
    #     return f"item_knn_{self.K}"
