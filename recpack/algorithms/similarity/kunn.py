import logging
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted

from recpack.util import get_top_K_values
from recpack.algorithms import Algorithm
from recpack.data.matrix import Matrix, to_csr_matrix
from recpack.algorithms.util import get_users, invert_np_array

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
        # To make sure the input is interpreted as a binary matrix
        X = to_csr_matrix(X, binary=True)

        self.training_interactions_ = csr_matrix(X, copy=True)  # Memoize X
        self.knn_i_ = self._fit_item_knn(X)

        return self

    def predict(self, X: Matrix) -> csr_matrix:
        """
        The prediction can easily be calculated by computing score matrix for users and add this to the already
        calculated score matrix for items.
        :param X: Sparse binary user-item matrix which will be used to do the predictions. The scores will returned for
        the users of this input matrix.
        :return: User-item matrix with the prediction scores as values.
        """
        check_is_fitted(self)

        X = to_csr_matrix(X, binary=True)

        knn_u = self._fit_user_knn(X)

        users_to_predict = set(X.nonzero()[0])
        num_users = X.shape[0]

        # Combine the memoized training interactions with the predict interactions
        # We will only use this combination for the user we are trying to predict for!
        Combined = self._union_csr_matrices(self.training_interactions_, X)

        # Cu rooted is based on the combined interactions
        _, Cu_rooted, _ = self._calculate_scaled_matrices(Combined)
        # Ci rooted is based on only the training interactions.
        _, _, Ci_rooted = self._calculate_scaled_matrices(self.training_interactions_)

        # Element-wise multiplication with Ci_rooted. Result is training_ui / sqrt(c_i).
        TrXi = csr_matrix(self.training_interactions_.multiply(Ci_rooted))
        # Element-wise multiplication. Result is combined_ui / sqrt(c.T_u)
        CombinedCu = csr_matrix(Combined.multiply(Cu_rooted.T))

        # Matrix multiplication. Result is knn_u * CombinedCi
        Su = csr_matrix(knn_u * TrXi)

        # Matrix multiplication. Result is CombinedCu * knn_i
        Si = csr_matrix(CombinedCu * self.knn_i_)

        self.S_ = Si + Su
        self.Su_ = Su
        self.Si_ = Si

        # The scores for the target users are calculated
        # by multiplying a diagonal matrix by the score matrix. On the
        # diagonal there is a 1 if that user is a target user.
        pred_users = self._get_users_diag(users_to_predict, num_users)
        scores = csr_matrix(pred_users * self.S_)
        self._check_prediction(scores, X)

        return scores

    def _calculate_scaled_matrices(
        self, X: csr_matrix
    ) -> (csr_matrix, np.ndarray, np.ndarray):
        """
        Helper method to calculate the scaled matrices. First the scaled matrix of X will be calculated base on each
        user and item counts. Ci-rooted is a vector with the number of users for each item.
        For these values, 1 divided by the square root will be taken. And vice versa for the Cu-rooted.
        :param X: Sparse binary user-item matrix
        :return: 3-tuple of a csr_matrix and ndarrays will be returned: Xscaled, Cu_rooted, Ci_rooted.
        """
        Cu_rooted = self._get_rooted_counts(X, 1)
        Ci_rooted = self._get_rooted_counts(X, 0)

        # Element-wise multiplication. Result is x_ui_scaled = x_ui / (sqrt(c_u) * sqrt(c_i) )
        Xscaled = csr_matrix(X.multiply(Cu_rooted.T).multiply(Ci_rooted))

        return Xscaled, Cu_rooted, Ci_rooted

    def _get_rooted_counts(self, X: csr_matrix, axis: int) -> np.ndarray:
        """
        Helper method to determine the Cu- and Ci_rooted vectors.
        :param X: Sparse binary user-item matrix
        :param axis: If 0, the Ci_rooted will be calculated and for Cu_rooted axis 1 must be used.
        :return: Vector with the rooted counts depending on the axis.
        """
        Cx = np.asarray(X.sum(axis=axis).flatten())
        Cx_calc = 1.0 / np.sqrt(Cx)
        Cx_calc[Cx_calc >= np.inf] = 0

        return Cx_calc

    def _create_mask(self, users_all: set, users: set, shape: tuple) -> csr_matrix:
        """
        Helper method to create a mask matrix. It is used to filter out the similarities between target users.
        :param users_all: Set of all combined users.
        :param users: Set of target users.
        :param shape: Shape of the user similarity matrix.
        :return: Sparse matrix with 1's where the users are linked to target users.
        """
        num_users_predict = len(users)
        pred_users = list(users)

        U_x = []
        U_y = []
        V = []
        for u in users_all:
            U_x += [u] * num_users_predict
            U_y += pred_users
            V += [1] * num_users_predict

        mask = csr_matrix((V, (U_x, U_y)), shape=shape)

        return mask

    def _get_users_diag(self, users: set, num_users: int) -> csr_matrix:
        """
        Predict for target users, diagonal matrix with 1's when the prediction scores need to be computed for these
        users.
        :param users: Set of target users.
        :param num_users: Total number of user interactions.
        :return: Sparse diagonal matrix.
        """
        V = np.ones((len(users),))
        U = np.array(list(users))

        return csr_matrix((V, (U, U)), shape=(num_users, num_users))

    def _fit_item_knn(self, X: csr_matrix) -> csr_matrix:
        """
        Helper method to compute the Item KNN, used in the KUNN implementation.
        @param X: Sparse binary user-item matrix
        @return: Item KNN matrix for X
        """
        Xscaled, _, Ci_rooted = self._calculate_scaled_matrices(X)

        # Element-wise multiplication with Ci_rooted. Result is x.T_iu / sqrt(c_i).
        XtCi = csr_matrix(X.T.multiply(Ci_rooted.T))
        # Matrix multiplication. Result is XtCi * Xscaled
        sim_i = csr_matrix(XtCi * Xscaled)
        # Eliminate self-similarity
        sim_i.setdiag(0)

        return get_top_K_values(sim_i, self.Ki).T

    def _fit_user_knn(self, X: csr_matrix) -> csr_matrix:
        """
        Helper method to compute the User KNN, used in the KUNN implementation. The memoized training interactions are
        used to compute the user similarities.
        @param X: Sparse binary user-item matrix
        @return: User KNN matrix for X
        """
        users_to_predict = get_users(X)

        # Combine the memoized training interactions with the predict interactions
        combined_interactions = self._union_csr_matrices(self.training_interactions_, X)

        # Cut combined interactions to only nonzero users in prediction matrix.
        mask = np.zeros(combined_interactions.shape[0])
        mask[users_to_predict] = 1
        # Turn mask into a column vector
        mask = mask.reshape(mask.shape[0], 1)
        # Select the interactions for nonzero users in mask
        print(combined_interactions.shape, mask.shape)
        combined_interactions_selected_users = csr_matrix(
            combined_interactions.multiply(mask)
        )

        # Compute the interactions that are only in the prediction matrix.
        combined_interactions_only_predict = (
            combined_interactions_selected_users
            - self.training_interactions_.multiply(mask)
        )

        # Count the number of interactions per user for which we need to predict
        # This count is based on the union of train and predict data
        pred_user_interaction_counts = combined_interactions_selected_users.sum(axis=1)

        # Counts based on only training data
        train_user_counts = self.training_interactions_.sum(axis=1)
        train_item_counts = self.training_interactions_.sum(axis=0)

        # Compute the c(i) values in the paper
        # Because we have to account for items that occur both in train and predict,
        # but can only use interactions in the X matrix for the user we are computing
        # similarities for (avoid leakage of data),
        # we need to add 1 to the training counts in some occasions.
        #
        # We do this by taking the count in the training matrix per item.
        # vertically stacking these values to get these counts for each user
        # And we then add the interactions, that only occur in the prediction dataset,
        # for prediction users,
        #
        # This gives us per user the accurate count per item,
        # taking into account training data, and only their own history
        # from the prediction dataset.
        item_counts_per_user = (
            np.vstack([train_item_counts for _ in range(X.shape[0])])
            + combined_interactions_only_predict
        )

        # Similarities are computed by matrix multiplication of two interaction matrices
        # the training matrix is scaled by dividing each interaction by
        #   the square root of the number of user interactions.
        # The combined interactions for prediction users is scaled by dividing
        #   by the square root of user interactions
        #   and by the square root of the interactions with the item.
        # fmt:off
        similarities = (
            combined_interactions_selected_users.multiply(
                invert_np_array(np.sqrt(pred_user_interaction_counts))
            ).multiply(
                invert_np_array(np.sqrt(item_counts_per_user))
            )
            @
            self.training_interactions_.multiply(
                invert_np_array(np.sqrt(train_user_counts))
            ).T
        )
        # fmt:on

        similarities.setdiag(0)

        return get_top_K_values(similarities, self.Ku)

    def _union_csr_matrices(self, a: csr_matrix, b: csr_matrix) -> csr_matrix:
        """
        Helper method to combine 2 binary csr_matrices.
        @param a: Binary csr_matrix
        @param b: Binary csr_matrix
        @return: The union of a and b
        """
        # pre-condition: a and b should be a binary csr_matrix (the values only consist out of 1.0's)
        return csr_matrix(a.astype(np.bool) + b.astype(np.bool)) * 1.0
