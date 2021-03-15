import numpy as np
import scipy.sparse
import sklearn.decomposition

from recpack.algorithms.base import (
    FactorizationAlgorithm,
    ItemSimilarityMatrixAlgorithm,
)
from recpack.data.matrix import Matrix, to_csr_matrix


class NMF(FactorizationAlgorithm):
    """Non negative matrix factorization.

    Matrix is decomposed into a user embedding and an item embedding of the same size.
    Decomposition happens using the NMF implementation from sklearn.
    Uses default parameters from NMF.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import NMF

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        algo = NMF(num_components=2)
        # Fit algorithm
        algo.fit(X)

        # After fitting user and item embeddings are available
        print(algo.user_features_.shape)
        # (3, 2)
        print(algo.item_features_)
        # (3, 2)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param num_components: The size of the latent dimension
    :type num_components: int

    :param random_state: The seed for the random state to allow for comparison,
                            defaults to 42
    :type random_state: int, optional
    :param alpha: regularization parameter, defines how much regularization is applied.
    :type alpha: float, optional
    :param l1_ratio: Defines how much L1 normalisation is used,
        compared to L2 normalisation.
        Value between 1 and 0, where 1 means only L1 normalisation,
        and 0 only L2 normalisation.
        Defaults to 0
    :type l1_ratio: float, optional
    """

    def __init__(self, num_components=100, random_state=42, alpha=0.0, l1_ratio=0.0):
        super().__init__(num_components)
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _fit(self, X: scipy.sparse.csr_matrix):

        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(
            n_components=self.num_components,
            init="random",
            random_state=self.random_state,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
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

    :param random_state: The seed for the random state to allow for comparison,
                            defaults to 42
    :type random_state: int, optional
    """

    def __init__(self, num_components=100, random_state=42):
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def _fit(self, X: scipy.sparse.csr_matrix):
        self.model_ = NMF(self.num_components, self.random_state)
        self.model_.fit(X)

        self.similarity_matrix_ = (
            self.model_.item_features_.T @ self.model_.item_features_
        )
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)


class SVD(FactorizationAlgorithm):
    """Singular Value Decomposition as dimension reduction recommendation algorithm.

    SVD computed using the TruncatedSVD implementation from sklearn.
    U x Sigma x V = X
    U are the user features, and the item features are computed as Sigma x V.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import SVD

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        algo = SVD(num_components=2)
        # Fit algorithm
        algo.fit(X)

        # After fitting user and item embeddings are available
        print(algo.user_features_.shape)
        # (3, 2)
        print(algo.item_features_)
        # (3, 2)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()


    :param num_components: The size of the latent dimension
    :type num_components: int

    :param random_state: The seed for the random state to allow for comparison
    :type random_state: int
    """

    def __init__(self, num_components=100, random_state=42):
        super().__init__(num_components=num_components)

        self.random_state = random_state

    def _fit(self, X: scipy.sparse.csr_matrix):

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

    :param random_state: The seed for the random state to allow for comparison,
                            defaults to 42
    :type random_state: int, optional
    """

    def __init__(self, num_components=100, random_state=42):
        super().__init__()
        self.num_components = num_components
        self.random_state = random_state

    def _fit(self, X: scipy.sparse.csr_matrix):
        self.model_ = SVD(self.num_components, self.random_state)
        self.model_.fit(X)

        self.similarity_matrix_ = (
            self.model_.item_features_.T @ self.model_.item_features_
        )
        # Remove self similarity.
        np.fill_diagonal(self.similarity_matrix_, 0)
