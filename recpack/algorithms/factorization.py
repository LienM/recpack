import numpy as np
import scipy.sparse
import sklearn.decomposition

from recpack.algorithms.base import Algorithm, ItemSimilarityMatrixAlgorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class FactorizationAlgorithm(Algorithm):
    """Base class for Factorization algorithms.

    A factorization algorithm fits a user feature and item feature matrix.

    Prediction happens by multiplying a users features with the item features.

    TODO -> Add info for creating your own factorization algorithm
    TODO -> In the Neural Network we call things embeddings,
        probably should call the features here embeddings?

    :param num_components: the dimension of the feature matrices. defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_components=100):
        super().__init__()
        self.num_components = num_components

    def _predict(self, X: Matrix):
        assert X.shape == (self.user_features_.shape[0], self.item_features_.shape[1])
        users = list(set(X.nonzero()[0]))
        result = scipy.sparse.lil_matrix(X.shape)
        result[users] = self.user_features_[users] @ self.item_features_
        return result.tocsr()


class NMF(FactorizationAlgorithm):
    """Non negative matrix factorization.

    Matrix is decomposed into a user feature matrix, and a

    :param FactorizationAlgorithm: [description]
    :type FactorizationAlgorithm: [type]
    :return: [description]
    :rtype: [type]
    """

    # TODO check params NMF to see which ones are useful.
    def __init__(self, num_components=100, random_state=42):
        """NMF factorization implemented using the sklearn library.
        :param num_components: The size of the latent dimension
        :type num_components: int

        :param random_state: The seed for the random state to allow for comparison,
                             defaults to 42
        :type random_state: int, optional
        """
        super().__init__(num_components)
        self.num_components = num_components
        self.random_state = random_state

    def _fit(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(
            n_components=self.num_components,
            init="random",
            random_state=self.random_state,
        )

        # Factorization is W * H. Where W contains user latent vectors, and H
        # contains item latent vectors
        self.user_features_ = model.fit_transform(X)
        self.item_features_ = model.components_

        # Post conditions
        assert self.user_features_.shape == (X.shape[0], self.num_components)
        assert self.item_features_.shape == (self.num_components, X.shape[1])

        return self


class NMFItemToItem(ItemSimilarityMatrixAlgorithm):
    """Algorithm using similarities between NMF item embeddings.

    Item embeddings are computed using the NMF algorithm,
    then a similarity matrix is constructed by
    computing the dot product between the embeddings.

    :param num_components: The size of the latent dimension
    :type num_components: int

    :param random_state: The seed for the random state to allow for comparison,
                            defaults to 42
    :type random_state: int, optional
    """

    def __init__(self, num_components=100, random_state=42):
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def _fit(self, X: Matrix):
        self.model_ = NMF(self.num_components, self.random_state)
        self.model_.fit(X)

        self.similarity_matrix_ = (
            self.model_.item_features_.T @ self.model_.item_features_
        )
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)


class SVD(FactorizationAlgorithm):
    """Singular Value Decomposition as dimension reduction recommendation algorithm.

    :param num_components: The size of the latent dimension
    :type num_components: int

    :param random_state: The seed for the random state to allow for comparison
    :type random_state: int
    """

    def __init__(self, num_components=100, random_state=42):
        super().__init__()

        self.num_components = num_components
        self.random_state = random_state

    def _fit(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        # TODO use other parameter options?
        model = sklearn.decomposition.TruncatedSVD(
            n_components=self.num_components, n_iter=7, random_state=self.random_state
        )
        # Factorization computes U x Sigma x V
        # U are the user features,
        # Sigma x V are the item features.
        self.user_features_ = model.fit_transform(X)

        V = model.components_
        sigma = scipy.sparse.diags(model.singular_values_)
        self.item_features_ = sigma @ V

        # Post conditions
        assert self.user_features_.shape == (X.shape[0], self.num_components)
        assert self.item_features_.shape == (self.num_components, X.shape[1])

        return self


class SVDItemToItem(ItemSimilarityMatrixAlgorithm):
    def __init__(self, num_components=100, random_state=42):
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def _fit(self, X: Matrix):
        self.model_ = SVD(self.num_components, self.random_state)
        self.model_.fit(X)

        self.similarity_matrix_ = (
            self.model_.item_features_.T @ self.model_.item_features_
        )
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)
