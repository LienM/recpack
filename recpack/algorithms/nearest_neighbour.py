# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import warnings
from typing import Optional

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.util import invert, to_binary
from recpack.util import get_top_K_values


def compute_conditional_probability(X: csr_matrix, pop_discount: float = 0) -> csr_matrix:
    """Compute conditional probability like similarity.

    Computation using equation (3) from the original ItemKNN paper.
    'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where :math:`\\mathbb{I}_{ui}` is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0,
    this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :param pop_discount: Parameter defining popularity discount. Defaults to 0
    :type pop_discount: float, Optional.
    """
    # matrix with co_mat_i,j =  SUM(1_u,i * X_u,j for each user u)
    # If the input matrix is binary, this is the cooccurence count matrix.
    co_mat = to_binary(X).T @ X

    # Compute the inverse of the item frequencies
    A = invert(diags(to_binary(X).sum(axis=0).A[0]).tocsr())

    if pop_discount:
        # This has all item similarities
        # Co_mat is weighted by both the frequencies of item i
        # and the frequency of item j to the pop_discount power.
        # If pop_discount = 1, this similarity is symmetric again.
        item_cond_prob_similarities = A @ co_mat @ A.power(pop_discount)
    else:
        # Weight the co_mat with the amount of occurences of item i.
        item_cond_prob_similarities = A @ co_mat

    # Set diagonal to 0, because we don't support self similarity
    item_cond_prob_similarities.setdiag(0)

    return item_cond_prob_similarities


def compute_cosine_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the cosine similarity between the items in the matrix.

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :return: similarity matrix
    :rtype: csr_matrix
    """
    # X.T otherwise we are doing a user KNN
    item_cosine_similarities = cosine_similarity(X.T, dense_output=False)
    item_cosine_similarities.setdiag(0)
    # Set diagonal to 0, because we don't want to support self similarity

    return item_cosine_similarities


def compute_pearson_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the pearson correlation as a similarity between each item in the matrix.

    Self similarity is removed.
    When computing similarity, the avg of nonzero entries per user is used.

    :param X: Rating or psuedo rating matrix.
    :type X: csr_matrix
    :return: similarity matrix.
    :rtype: csr_matrix
    """

    if (X == 1).sum() == X.nnz:
        raise ValueError("Pearson similarity can not be computed on a binary matrix.")

    count_per_item = (X > 0).sum(axis=0).A

    avg_per_item = X.sum(axis=0).A.astype(float)

    avg_per_item[count_per_item > 0] = avg_per_item[count_per_item > 0] / count_per_item[count_per_item > 0]

    X = X - (X > 0).multiply(avg_per_item)

    # Given the rescaled matrix, the pearson correlation is just cosine similarity on this matrix.
    return compute_cosine_similarity(X)


class ItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Item K Nearest Neighbours model.

    First described in 'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis,
    ACM Transactions on Information Systems (TOIS) 22.1 (2004): 143-177

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"`` and ``"conditional_probability"``

    Cosine similarity between item i and j is computed as

    .. math::
        sim(i,j) = \\frac{X_i X_j}{||X_i||_2 ||X_j||_2}

    The conditional probablity based similarity of item i with j is computed as

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where I_ui is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0, this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    If sim_normalize is True, the scores are normalized per predictive item,
    making sure the sum of each row in the similarity matrix is 1.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator,
        to discount contributions of very popular items.
        Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: float, optional
    :param normalize_X: Normalize rows in the interaction matrix so that
        the contribution of users who have viewed more items is smaller,
        defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to
        counteract artificially large similarity scores when the predictive item is
        rare, defaults to False.
    :type normalize_sim: bool, optional
    :raises ValueError: If an unsupported similarity measure is passed.
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]
    """The supported similarity options"""

    def __init__(
        self,
        K=200,
        similarity: str = "cosine",
        pop_discount: Optional[float] = None,
        normalize_X: bool = False,
        normalize_sim: bool = False,
    ):
        super().__init__(K)

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

        if self.similarity != "conditional_probability" and pop_discount:
            warnings.warn(
                "Argument pop_discount is incompatible with all similarity \
                functions except conditional probability. \
                This argument will be ignored, \
                popularity discounting won't be applied.",
                UserWarning,
            )

        if type(pop_discount) == float and (pop_discount < 0 or pop_discount > 1):
            raise ValueError("Invalid value for pop_discount. Value should be between 0 and 1.")

        self.pop_discount = pop_discount

        self.normalize_X = normalize_X
        # Sim_normalize takes precedence.
        self.normalize_sim = normalize_sim

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""

        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X, self.pop_discount)

        item_similarities = get_top_K_values(item_similarities, K=self.K)

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

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator,
        to discount contributions of very popular items.
        Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: float, optional
    :param normalize_X: Normalize rows in the interaction matrix so that
        the contribution of users who have viewed more items is smaller,
        defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to
        counteract artificially large similarity scores
        when the predictive item is rare,
        defaults to False.
    :type normalize_sim: bool, optional
    :param pdf: Which probability distribution to use,
        can be one of ["empirical", "uniform", "softmax_empirical"],
        defaults to "empirical"
    :type pdf: str, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :raises ValueError: If an unsupported similarity measure or
        probability distribution is passed.
    """

    SUPPORTED_SAMPLING_FUNCTIONS = ["empirical", "uniform", "softmax_empirical"]
    """The supported similarity options"""

    def __init__(
        self,
        K=200,
        similarity: str = "cosine",
        pop_discount: Optional[float] = None,
        normalize_X: bool = False,
        normalize_sim: bool = False,
        pdf: str = "empirical",
        seed: Optional[int] = None,
    ):
        super().__init__(
            K=K,
            similarity=similarity,
            pop_discount=pop_discount,
            normalize_X=normalize_X,
            normalize_sim=normalize_sim,
        )

        if pdf not in self.SUPPORTED_SAMPLING_FUNCTIONS:
            raise ValueError(f"Sampling function {pdf} not supported")

        self.pdf = pdf

        if seed is None:
            seed = np.random.get_state()[1][0]

        np.random.seed(seed)
        self.seed = seed

    def _compute_pdf(self, pdf: str, sim_matrix: csr_matrix) -> np.ndarray:
        # TODO Outside of the class maybe?
        sim_matrix = sim_matrix.toarray()
        if pdf == "empirical":
            # Add the None dimension at the end to do a row-wise division.
            # Otherwise the default is column-wise.
            p = sim_matrix / sim_matrix.sum(axis=1)[:, None]
        elif pdf == "uniform":
            p = np.ones(sim_matrix.shape) / sim_matrix.shape[1]
        elif pdf == "softmax_empirical":
            softmax_item_sims = np.exp(sim_matrix)
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
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X, self.pop_discount)

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
    """Select K values random values for every row in X,
    sampled according to the probabilities in pdf.
    All other values in the row are set to zero.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Amount of values to select.
    :type K: int
    :param pdf: np.ndarray of probabilities of items in X, given another item.
        Rows should sum to 1.
    :type pdf: np.ndarray
    :return: Matrix with K values per row.
    :rtype: csr_matrix
    """
    items = np.arange(0, X.shape[1], dtype=int)

    U, I, V = [], [], []

    for row_ix in range(0, X.shape[0]):
        # Select one more, so that we can eliminate the item itself.
        selected_K = np.random.choice(items, size=K + 1, p=pdf[row_ix, :], replace=False)

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
