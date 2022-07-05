from typing import Optional
from scipy.sparse import csr_matrix, diags
import sklearn.decomposition

from recpack.algorithms.base import (
    FactorizationAlgorithm,
)


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
        print(algo.user_embedding_.shape)
        # (3, 2)
        print(algo.item_embedding_)
        # (3, 2)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param num_components: The size of the latent dimension
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison,
                            defaults to None
    :type seed: int, optional
    :param alpha: Regularization parameter, defines how much regularization is applied.
    :type alpha: float, optional
    :param l1_ratio: Defines how much L1 normalisation is used,
        compared to L2 normalisation.
        Value between 1 and 0, where 1 means only L1 normalisation,
        and 0 only L2 normalisation.
        Defaults to 0
    :type l1_ratio: float, optional
    """

    def __init__(
        self, num_components: int = 100, seed: Optional[int] = None, alpha: float = 0.0, l1_ratio: float = 0.0
    ):
        super().__init__(num_components)
        self.seed = seed
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _fit(self, X: csr_matrix):

        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(
            n_components=self.num_components,
            init="random",
            random_state=self.seed,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
        )

        # Factorization is W * H. Where W contains user latent vectors, and H
        # contains item latent vectors
        self.user_embedding_ = model.fit_transform(X)
        self.item_embedding_ = model.components_

        # Post conditions
        assert self.user_embedding_.shape == (X.shape[0], self.num_components)
        assert self.item_embedding_.shape == (self.num_components, X.shape[1])


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
        print(algo.user_embedding_.shape)
        # (3, 2)
        print(algo.item_embedding_)
        # (3, 2)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()


    :param num_components: The size of the latent dimension
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison, defaults to None
    :type seed: int, optional
    """

    def __init__(self, num_components: int = 100, seed: Optional[int] = None):
        super().__init__(num_components=num_components)

        self.seed = seed

    def _fit(self, X: csr_matrix):

        model = sklearn.decomposition.TruncatedSVD(n_components=self.num_components, n_iter=7, random_state=self.seed)
        # Factorization computes U x Sigma x V
        # U are the user features,
        # Sigma x V are the item features.
        self.user_embedding_ = model.fit_transform(X)

        V = model.components_
        sigma = diags(model.singular_values_)
        self.item_embedding_ = sigma @ V

        # Post conditions
        assert self.user_embedding_.shape == (X.shape[0], self.num_components)
        assert self.item_embedding_.shape == (self.num_components, X.shape[1])
