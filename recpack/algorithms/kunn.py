# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
import numpy as np

from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms import Algorithm
from recpack.algorithms.util import (
    get_users,
    invert,
    union_csr_matrices,
)
from recpack.util import get_top_K_values

logger = logging.getLogger("recpack")


class KUNN(Algorithm):
    """Unified Nearest Neighbour algorithm combining user and item neighbourhood methods.

    KUNN Algorithm as described in 'Unifying Nearest Neighbors Collaborative Filtering'
    Verstrepen et al. (10.1145/2645710.2645731)

    Computes the item KNN model and stores training interactions at fitting time.
    Computes the user KNN model between test and training users at prediction time.

    Scores are computed as a sum of item and user similarity.

    user KNN are computed using

    .. math::

        sim(u,v) = \\sum_{i \\in I} { \\frac{ R_{ui} R_{vi}}{\\sqrt{c(u) c(v) c(i)}}}

    item KNN are computed as


    .. math::

        sim(i,j) = \\sum_{u \\in U} { \\frac{ R_{ui} R_{vi}}{\\sqrt{c(i) c(u) c(j)}}}

    Similarity is computed as

    .. math::

        sim(u, i) = S_U(u, i) + S_I(u, i)

    Where user similarity is computed as

    .. math::

        S_U(u, i) = \\sum_{v \\in KNN(u)} \\frac{R_{vi} * sim(u,v)}{\\sqrt{c(i)}}

    and item similarity is computed as

    .. math::

        S_I(u, i) = \\sum_{j \\in KNN(i)} \\frac{R_{uj} * sim(i, j)}{\\sqrt{c(u)}}



    :param Ku: How many neighbours to keep in the user similarity matrix.
        Defaults to 100.
    :type Ku: int, optional
    :param Ki: How many items to keep as neighbours in the item similarity matrix.
        Defaults to 100.
    :type Ki: int, optional

    """

    def __init__(self, Ku: int = 100, Ki: int = 100):
        super().__init__()
        self.Ku = Ku
        self.Ki = Ki

    def _fit(self, X: csr_matrix):
        """Calculate the item similarity matrix based on the interactions.

        :param X: Sparse binary user-item interaction matrix
            which will be used to fit the algorithm.
        """

        self.training_interactions_ = csr_matrix(X, copy=True)
        self.knn_i_ = self._fit_item_knn(X)

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict recommendations for all nonzero users in the interaction matrix.

        Computes a userKNN model, and then predicts based on the combined
        score of user and item similarity.

        :param X: Sparse binary user-item matrix which will be used as history.
        :type X: csr_matrix
        :return: User-item matrix with the prediction scores as values.
        :rtype: csr_matrix
        """

        # Memorised training interactions are used in `_fit_user_knn` as well
        knn_u = self._fit_user_knn(X)

        users_to_predict = get_users(X)

        # Combine the memoized training interactions with the predict interactions
        # We will only use this combination for the user we are trying to predict for!
        combined_interactions = union_csr_matrices(self.training_interactions_, X)

        # Compute user similarity,
        # Formula (10) in paper.
        # we'll pull the 1/ sqrt(c(u)c(i))
        # into the subcomputations of user similarity and item similarity.

        # Note that user_knn scores already compute the
        # inner sum in the user similarity formula
        # if we also include the division by sqrt(u)
        #
        # So we can compute user similarity as
        # KNN(u) @ (training_interactions / sqrt(count(i)))
        # Which will count occurrences of the target item,
        # weighted by the user similarity
        # and 1 / sqrt(count(i))
        item_counts = self.training_interactions_.sum(axis=0)

        user_similarity = csr_matrix(knn_u @ self.training_interactions_.multiply(invert(np.sqrt(item_counts))))

        # Compute item similarities
        # Similar trick, 1/sqrt(c(i)) is already included in the item KNN computation.
        # Which computes then computes the inner sum.
        # The score is then generated by multiplying the combined matrix with the
        # Item KNN scores
        # And dividing by sqrt(c(u)), the square root of the user's interactions.
        user_counts = combined_interactions.sum(axis=1)
        item_similarity = csr_matrix(combined_interactions.multiply(invert(np.sqrt(user_counts))) @ self.knn_i_)

        similarity = item_similarity + user_similarity

        # We only need to set the similarities for users which we need to predict for
        # TODO: There is probably a way to optimise computation by not needing to
        # compute the similarities, similar to computation of user_knn.

        scores = lil_matrix(X.shape)
        scores[users_to_predict] = similarity[users_to_predict]

        scores = scores.tocsr()

        return scores

    def _fit_item_knn(self, X: csr_matrix) -> csr_matrix:
        """
        Helper method to compute the Item KNN, used in the KUNN implementation.

        """

        user_counts = X.sum(axis=1)
        item_counts = X.sum(axis=0)

        item_to_item_similarity = X.multiply(invert(np.sqrt(user_counts))).multiply(
            invert(np.sqrt(item_counts))
        ).T @ X.multiply(invert(np.sqrt(item_counts)))

        # Eliminate self-similarity
        item_to_item_similarity.setdiag(0)

        return get_top_K_values(item_to_item_similarity, self.Ki).T

    def _fit_user_knn(self, X: csr_matrix) -> csr_matrix:
        """Helper method to compute the User KNN, used in the KUNN implementation.
        The memoized training interactions are used to compute the user similarities.

        """

        users_to_predict = get_users(X)

        # Combine the memoized training interactions with the predict interactions
        combined_interactions = union_csr_matrices(self.training_interactions_, X)

        # Cut combined interactions to only nonzero users in prediction matrix.
        mask = np.zeros(combined_interactions.shape[0])
        mask[users_to_predict] = 1
        # Turn mask into a column vector
        mask = mask.reshape(mask.shape[0], 1)
        # Select the interactions for nonzero users in mask
        combined_interactions_selected_users = csr_matrix(combined_interactions.multiply(mask))

        # Compute the interactions that are only in the prediction matrix.
        combined_interactions_only_predict = (
            combined_interactions_selected_users - self.training_interactions_.multiply(mask)
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
            np.vstack([train_item_counts for _ in range(X.shape[0])]) + combined_interactions_only_predict
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
                invert(np.sqrt(pred_user_interaction_counts))
            ).multiply(
                invert(np.sqrt(item_counts_per_user))
            )
            @
            self.training_interactions_.multiply(
                invert(np.sqrt(train_user_counts))
            ).T
        )
        # fmt:on

        similarities.setdiag(0)

        return get_top_K_values(similarities, self.Ku)
