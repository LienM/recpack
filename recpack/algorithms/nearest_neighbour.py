import warnings
from typing import List, Optional

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csr import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.util import invert
from recpack.util import get_top_K_values


class ItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Item K Nearest Neighbours model.

    First described in 'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis,
    ACM Transactions on Information Systems (TOIS) 22.1 (2004): 143-177

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"`` and ``"conditional_probability"``

    - Cosine similarity between item i and j is computed as
      the ``count(i and j) / (count(i)*count(j))``.
    - Conditional probablity of item i with j is computed
      as ``count(i and j) / (count(i))``.
      Note that this is a non-symmetric similarity measure.

    If sim_normalize is True, the scores are normalized per predictive item,
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
        algo = ItemKNN(K=2, similarity='conditional_probability', sim_normalize=True)
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
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator, to discount contributions
        of very popular items. Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: Optional[float], optional
    :param normalize_X: Normalize rows in the interaction matrix so that the contribution of
        users who have viewed more items is smaller, defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to counteract
        artificially large similarity scores when the predictive item is rare, defaults to False.
    :type normalize_sim: bool, optional
    :param normalize: DEPRECATED. Use normalize_sim instead.
        Defaults to False
    :type normalize: bool, optional
    :raises ValueError: If an unsupported similarity measure is passed.
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]
    """The supported similarity options"""

    def __init__(self, K=200, similarity: str = "cosine", pop_discount: Optional[float] = None,
                 normalize_X: bool = False, normalize_sim: bool = False, normalize: bool = False):
        super().__init__(K)

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

        if self.similarity != "conditional_probability" and pop_discount:
            warnings.warn("Argument pop_discount is incompatible with all similarity \
                functions except conditional probability. This argument will be ignored, \
                popularity discounting won't be applied.", UserWarning)

        if type(pop_discount) == float and (pop_discount < 0 or pop_discount > 1):
            raise ValueError(
                "Invalid value for pop_discount. Value should be between 0 and 1.")

        self.pop_discount = pop_discount

        if normalize:
            warnings.warn(
                "Use of argument normalize is deprecated. Use normalize_sim instead.", DeprecationWarning)

        self.normalize_X = normalize_X
        # Sim_normalize takes precedence.
        self.normalize_sim = normalize_sim if normalize_sim else normalize

        self.normalize = normalize

    def _compute_conditional_probability(self, X: csr_matrix) -> csr_matrix:
        # Cooccurence matrix
        co_mat = X.T @ X

        # Adding 1 additive smoothing to occurrences to avoid division by 0
        A = invert(diags(co_mat.diagonal()).tocsr())

        # We're trying to get a matrix S of P(j|i) where j is the column index,
        # i is the row index, so that we can later do X * S to obtain predictions.

        if self.pop_discount:
            # This has all item similarities
            item_cond_prob_similarities = A @ co_mat @ A.power(
                self.pop_discount)
        else:
            item_cond_prob_similarities = A @ co_mat
        # Set diagonal to 0, because we don't support self similarity
        item_cond_prob_similarities.setdiag(0)

        return item_cond_prob_similarities

    def _compute_cosine(self, X: csr_matrix) -> csr_matrix:
        # X.T otherwise we are doing a user KNN
        item_cosine_similarities = cosine_similarity(X.T, dense_output=False)
        item_cosine_similarities.setdiag(0)
        # Set diagonal to 0, because we don't want to support self similarity

        return item_cosine_similarities

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            item_similarities = self._compute_cosine(X)
        elif self.similarity == "conditional_probability":
            item_similarities = self._compute_conditional_probability(X)

        item_similarities = get_top_K_values(item_similarities, self.K)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities


class ItemPNN(ItemKNN):
    """Item Probabilistic Nearest Neighbours model.

    First described in Panagiotis Adamopoulos and Alexander Tuzhilin. 2014.
    'On over-specialization and concentration bias of recommendations:
    probabilistic neighborhood selection in collaborative filtering systems'.
    In Proceedings of the 8th ACM Conference on Recommender systems (RecSys '14).
    Association for Computing Machinery, New York, NY, USA, 153–160.
    DOI:https://doi.org/10.1145/2645710.2645752

    For each item K neighbours are selected either uniformly or based on the empirical
    distribution of the items (or a softmax thereof).
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"`` and ``"conditional_probability"``

    - Cosine similarity between item i and j is computed as
      the ``count(i and j) / (count(i)*count(j))``.
    - Conditional probablity of item i with j is computed
      as ``count(i and j) / (count(i))``.
      Note that this is a non-symmetric similarity measure.

    If sim_normalize is True, the scores are normalized per predictive item,
    making sure the sum of each row in the similarity matrix is 1.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import ItemKNN

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        # We'll only keep the closest neighbour for each item.
        # Default uses cosine similarity
        algo = ItemPNN(K=1, pdf="uniform")
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
        algo = ItemPNN(K=2, similarity='conditional_probability', sim_normalize=True, pdf='uniform')
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
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator, to discount contributions
        of very popular items. Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: Optional[float], optional
    :param normalize_X: Normalize rows in the interaction matrix so that the contribution of
        users who have viewed more items is smaller, defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to counteract
        artificially large similarity scores when the predictive item is rare, defaults to False.
    :type normalize_sim: bool, optional
    :param pdf: Which probability distribution to use,
        can be one of ["empirical", "uniform", "softmax_empirical"], defaults to "empirical"
    :type pdf: str, optional
    :raises ValueError: If an unsupported similarity measure or probability distribution is passed.
    """
    SUPPORTED_SAMPLING_FUNCTIONS = [
        "empirical", "uniform", "softmax_empirical"]
    """The supported similarity options"""

    def __init__(self, K=200, similarity: str = "cosine", pop_discount: Optional[float] = None, normalize_X: bool = False, normalize_sim: bool = False, pdf: str = "empirical"):

        super().__init__(K=K, similarity=similarity, pop_discount=pop_discount,
                         normalize_X=normalize_X, normalize_sim=normalize_sim)

        if pdf not in self.SUPPORTED_SAMPLING_FUNCTIONS:
            raise ValueError(f"Sampling function {pdf} not supported")

        # TODO Add a random seed to make results reproducable
        self.pdf = pdf

    def _compute_pdf(self, pdf: str, X: csr_matrix) -> np.ndarray:
        # TODO Outside of the class maybe?
        X = X.toarray()
        if pdf == "empirical":
            p = X / X.sum(axis=1)[:, None]
        elif pdf == "uniform":
            p = np.ones(X.shape) / X.shape[1]
        elif pdf == "softmax_empirical":
            softmax_item_sims = np.exp(X)
            p = softmax_item_sims / softmax_item_sims.sum(axis=1)[:, None]
        else:
            raise ValueError(f"Sampling function {pdf} not supported")

        return p

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""
        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            item_similarities = self._compute_cosine(X)
        elif self.similarity == "conditional_probability":
            item_similarities = self._compute_conditional_probability(X)

        self.pdf_ = self._compute_pdf(self.pdf, item_similarities)

        item_similarities = get_K_values(item_similarities, self.K, self.pdf_)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities

    # def _predict(self, X: csr_matrix) -> csr_matrix:
    #     pass


def get_K_values(X: csr_matrix, K: int, pdf: np.ndarray) -> csr_matrix:
    """Select K values random values for every row in X, sampled according to the probabilities in pdf.
    All other values in the row are set to zero.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Amount of values to select.
    :type K: int
    :param pdf: List of probabilities of every item in X. Should sum to 1.
    :type pdf: List[float]
    :return: Matrix with K values per row.
    :rtype: csr_matrix
    """
    items = np.arange(0, X.shape[1], dtype=int)

    U, I, V = [], [], []

    for row_ix in range(0, X.shape[0]):
        # Select one more, so that we can eliminate the item itself.
        selected_K = np.random.choice(
            items, size=K + 1, p=pdf[row_ix, :], replace=False)

        try:
            # Eliminate the item itself if it was selected.
            mismatch = np.where(selected_K == row_ix)[0][0]
        except IndexError:
            # If it was not selected, just eliminate the last item.
            mismatch = -1

        selected_K = np.delete(selected_K, mismatch)

        U.extend([row_ix] * K)
        I.extend(selected_K)
        V.extend([1] * K)

    data_K = csr_matrix((V, (U, I)), shape=X.shape)
    return data_K.multiply(X)
