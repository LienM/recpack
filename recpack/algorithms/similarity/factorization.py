import scipy.sparse
import sklearn.decomposition

from recpack.algorithms.similarity.base import (
    SimilarityMatrixAlgorithm,
)


class NMF(SimilarityMatrixAlgorithm):
    # TODO check params NMF to see which ones are useful.
    def __init__(self, n_components, random_state=42):
        """NMF factorization, where the item features
            are used to compute an item to item similarity matrix
            user features are not used, this avoids needing users as input.
        :param n_components: The size of the latent dimension
        :type n_components: int

        :param random_state: The seed for the random state to allow for comparison,
                             defaults to 42
        :type random_state: int, optional
        """
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X):
        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(
            n_components=self.n_components, init="random", random_state=0
        )

        # Factorization is W * H. Where W contains user latent vectors, and H
        # contains item latent vectors
        _ = model.fit_transform(X)
        H = model.components_

        # Compute an item to item similarity matrix by computing the dot product
        # between the 2 item latent vectors
        self.similarity_matrix_ = H.T @ H

        self._check_fit_complete()
        return self


class SVD(SimilarityMatrixAlgorithm):
    """Singular Value Decomposition as dimension reduction recommendation algorithm.

    User latent matrix is discarded, instead the item to item similarity is computed,
    based on the item's latent features. (By using dot product of the 2 vectors)

    :param n_components: The size of the latent dimension
    :type n_components: int

    :param random_state: The seed for the random state to allow for comparison
    :type random_state: int
    """

    def __init__(self, n_components, random_state=42):
        super().__init__()

        self.n_components = n_components
        self.random_state = 42

    def fit(self, X):
        # TODO use other parameter options?
        model = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, n_iter=7, random_state=self.random_state
        )
        model.fit(X)

        # Factorization computes U x Sigma x V

        # In this implementation we avoid having to store the user features,
        # and compute an item to item similarity matrix based on the SVD output
        # Item similarity is computed as (Sigma * V).T * (sigma * V)
        # The dot product between the item latent feature vectors.
        V = model.components_
        sigma = scipy.sparse.diags(model.singular_values_)
        V_s = sigma @ V
        self.similarity_matrix_ = V_s.T @ V_s

        self._check_fit_complete()
        return self
