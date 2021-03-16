import numpy as np
import scipy
from scipy.sparse import diags
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from recpack.data.matrix import Matrix, to_csr_matrix
from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm


class ItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Item K Nearest Neighbours model.

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"`` and ``"conditional_probability"``

    - Cosine similarity between item i and j is computed as
      the ``count(i and j) / (count(i)*count(j))``.
    - Conditional probablity of item i with j is computed
      as ``count(i and j) / (count(i))``.
      Note that this is a non-simetrical similarity measure.

    If normalize is True, the scores are normalized per center item,
    making sure the sum of each row in the similarity matrix is 1.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import ItemKNN

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        # We'll only keep the closest neighbour for each item.
        # Default uses cosine similarity
        algo = ItemKNN(K=1)
        # Fit algorithm
        algo.fit(X)

        # We can inspect the fitted model
        print(algo.similarity_matrix_.nnz)
        # 3

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    **Example with Conditional Probability**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import ItemKNN

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        # We'll only keep the closest neighbour for each item.
        # we set the similarity measure to conditional probability
        # And enable normalization
        algo = ItemKNN(K=2, similarity='conditional_probability', normalize=True)
        # Fit algorithm
        algo.fit(X)

        # We can inspect the fitted model
        print(algo.similarity_matrix_.nnz)
        # 6

        # Similarities were normalized, so each row in the similarity matrix
        # sums to 1
        print(algo.similarity_matrix_.sum(axis=1))
        # [[1], [1], [1]]

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param normalize: Normalize scores per row in the similarity matrix,
        defaults to False
    :type normalize: bool, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]
    """The supported Similarity options"""

    def __init__(self, K=200, normalize=False, similarity: str = "cosine"):
        super().__init__(K)
        self.normalize = normalize
        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

    def _compute_conditional_probability(self, X):
        # Cooccurence matrix
        co_mat = X.T @ X

        # Adding 1 additive smoothing to occurrences to avoid division by 0
        A = diags(1 / (co_mat.diagonal() + 1))

        # This has all item similarities
        item_cond_prob_similarities = A @ co_mat
        # Set diagonal to 0, because we don't support self similarity
        item_cond_prob_similarities.setdiag(0)

        return item_cond_prob_similarities

    def _compute_cosine(self, X):
        # X.T otherwise we are doing a user KNN
        item_cosine_similarities = cosine_similarity(X.T, dense_output=False)

        item_cosine_similarities.setdiag(0)
        # Set diagonal to 0, because we don't want to support self similarity

        return item_cosine_similarities

    def _fit(self, X: Matrix):
        """Fit a cosine similarity matrix from item to item"""
        X = to_csr_matrix(X, binary=True)

        if self.similarity == "cosine":
            item_similarities = self._compute_cosine(X)
        elif self.similarity == "conditional_probability":
            item_similarities = self._compute_conditional_probability(X)

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(
                np.argpartition(item_similarities.toarray(), -self.K)
            )
            for j in best_items_row[-self.K :]
        ]
        # Create a mask matrix which will be pointwise multiplied with the
        # similarity matrix.
        mask = scipy.sparse.csr_matrix(
            ([1 for i in range(len(indices))], (list(zip(*indices))))
        )

        item_similarities = item_similarities.multiply(mask)

        if self.normalize:
            # normalize such that sum per row = 1
            transformer = Normalizer(norm="l1")
            item_similarities = scipy.sparse.csr_matrix(
                transformer.transform(item_similarities)
            )

        self.similarity_matrix_ = item_similarities
