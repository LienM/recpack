import logging
import numpy as np

from scipy.sparse import csr_matrix, diags
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms import Algorithm
from recpack.data.matrix import Matrix, to_csr_matrix

logger = logging.getLogger("recpack")


class KUNN(Algorithm):
    """
    KUNN Algorithm by Koen Verstrepen and Bart Goethals
    as described in paper 'Unifying Nearest Neighbors Collaborative Filtering' (10.1145/2645710.2645731)
    """
    def __init__(self, Ku: int = 100, Ki: int = 100):
        """
        Initialize the KUNN Algorithm but setting the number of K (top-K) for users and items.
        :param Ku: The top-K users which should be fit and predict for.
        :param Ki: The top-K items which should be fit and predict for.
        """
        super().__init__()
        self.Ku = Ku
        self.Ki = Ki

    def fit(self, X: Matrix) -> Algorithm:
        """
        Calculate the score matrix for items based on the binary matrix .
        :param X: Sparse binary user-item matrix which will be used to fit the algorithm.
        :return: The fitted KUNN Algorithm itself.
        """
        X = to_csr_matrix(X, binary=True)
        Xscaled, Cu_rooted, Ci_rooted = self._calculate_scaled_matrices(X)

        sim_i = csr_matrix((X.T * Xscaled) @ Ci_rooted)
        sim_i.setdiag(0)

        knn_i = self._get_top_K(sim_i, self.Ki)
        self.Si_ = csr_matrix((Cu_rooted * X) @ knn_i)

        return self

    def predict(self, X: Matrix, user_ids: np.array = None) -> csr_matrix:
        """
        The prediction can easily be calculated by computing score matrix for users and add this to the already
        calculated score matrix for items.
        :param X: Sparse binary user-item matrix which will be used to do the predictions. The scores will returned for
        the users of this input matrix.
        :param user_ids: Unused parameter.
        :return: User-item matrix with the prediction scores as values.
        """
        check_is_fitted(self)

        X = to_csr_matrix(X, binary=True)
        Xscaled, Cu_rooted, Ci_rooted = self._calculate_scaled_matrices(X)

        sim_u = csr_matrix((X * Xscaled.T) @ Cu_rooted)
        sim_u.setdiag(0)

        knn_u = self._get_top_K(sim_u, self.Ku)
        Su = csr_matrix((knn_u * X) @ Ci_rooted)

        self.S_ = self.Si_ + Su

        # Predict for users, diagnal matrix with 1's when the prediction scores need to be computed for these users.
        users = set(X.nonzero()[0])
        V = np.ones((len(users),))
        U = np.array(list(users))

        diag_users = csr_matrix((V, (U, U)), shape=(X.shape[0], X.shape[0]))

        scores = csr_matrix(diag_users * self.S_)
        self._check_prediction(scores, X)

        return scores

    def _get_top_K(self, X: csr_matrix, k: int) -> csr_matrix:
        """
        Return csr_matrix of top K item ranks for every user/item. The values of this matrix are the data values itself.
        :param X: The similarity matrix for users/items for calculating the KNN.
        :param k: Parameter for which the top K needs to be calculated.
        :return: Sparse matrix containing data values of top KNN sorted by rank.
        """
        U, I, V = [], [], []
        for row_ix, (le, ri) in enumerate(
                zip(X.indptr[:-1], X.indptr[1:])):
            K_row_pick = min(k, ri - le)

            if K_row_pick != 0:

                top_k_row = X.indices[
                    le
                    + np.argpartition(X.data[le:ri], list(range(-K_row_pick, 0)))[
                        -K_row_pick:
                    ]
                ]

                for rank, col_ix in enumerate(reversed(top_k_row)):
                    U.append(row_ix)
                    I.append(col_ix)
                    V.append(X[row_ix, col_ix])  # set value to the actual value to get the KNN

        return csr_matrix((V, (U, I)), shape=X.shape)

    def _calculate_scaled_matrices(self, X: csr_matrix) -> (csr_matrix, csr_matrix, csr_matrix):
        """
        Helper method to calculate the scaled matrices. First the scaled matrix of X will be calculated base on each
        user and item counts. Ci-rooted is a diagonal matrices with the number of users for each item on the diagonal.
        For these values, 1 divided by the square root will be taken. And vice versa for the Cu-rooted.
        :param X: Sparse binary user-item matrix
        :return: 3-tuple of csr_matrices will be returned: Xscaled, Cu_rooted, Ci_rooted.
        """
        Cu_rooted = self._get_rooted_counts(X, 1)
        Ci_rooted = self._get_rooted_counts(X, 0)

        Xscaled = (Cu_rooted @ X) @ Ci_rooted

        return Xscaled, Cu_rooted, Ci_rooted

    def _get_rooted_counts(self, X: csr_matrix, axis: int) -> csr_matrix:
        """
        Helper function to determine the Cu- and Ci_rooted diagonal matrices.
        :param X: Sparse binary user-item matrix
        :param axis: If 0, the Ci_rooted will be calculated and for Cu_rooted axis 1 must be used.
        :return: Diagonal matrix with
        """
        Cx = np.asarray(X.sum(axis=axis).flatten()).reshape(-1)
        Cx_calc = 1.0 / np.sqrt(Cx)
        Cx_calc[Cx_calc >= np.inf] = 0

        return diags(Cx_calc, 0)
