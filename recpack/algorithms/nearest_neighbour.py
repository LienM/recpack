import numpy as np
import scipy
from scipy.sparse import diags
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from recpack.data.matrix import Matrix, to_csr_matrix
from recpack.algorithms.base import TopKSimilarityMatrixAlgorithm


class ItemKNN(TopKSimilarityMatrixAlgorithm):
    def __init__(self, K=200, normalize=False, similarity: str = "cosine"):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item.

        If normalize is True, the scores are normalized per item.
        """
        super().__init__(K)
        self.normalize = normalize
        self.similarity = similarity

    def _compute_conditional_probability(self, X):
        co_mat = X.T @ X
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        A = diags(1 / co_mat.diagonal())

        # This has all item-cosine similarities. Now we should probably set N-K
        # to zero
        item_cosine_similarities = A @ co_mat

        # Set diagonal to 0, because we don't support self similarity
        item_cosine_similarities.setdiag(0)

        return item_cosine_similarities

    def _compute_cosine(self, X):
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        # X.T otherwise we are doing a user KNN
        item_cosine_similarities = cosine_similarity(X.T, dense_output=False)

        item_cosine_similarities.setdiag(0)
        # Set diagonal to 0, because we don't want to support self similarity

        return item_cosine_similarities

    def _fit(self, X: Matrix):
        """Fit a cosine similarity matrix from item to item"""
        X = to_csr_matrix(X, binary=True)

        if self.similarity == "cosine":
            item_cosine_similarities = self._compute_cosine(X)
        elif self.similarity == "conditional_probability":
            item_cosine_similarities = self._compute_conditional_probability(X)

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(
                np.argpartition(item_cosine_similarities.toarray(), -self.K)
            )
            for j in best_items_row[-self.K :]
        ]
        # Create a mask matrix which will be pointwise multiplied with the
        # similarity matrix.
        mask = scipy.sparse.csr_matrix(
            ([1 for i in range(len(indices))], (list(zip(*indices))))
        )

        item_cosine_similarities = item_cosine_similarities.multiply(mask)

        if self.normalize:
            # normalize per row
            row_sums = item_cosine_similarities.sum(axis=1)
            item_cosine_similarities = item_cosine_similarities / row_sums
            item_cosine_similarities = scipy.sparse.csr_matrix(item_cosine_similarities)

        self.similarity_matrix_ = item_cosine_similarities
