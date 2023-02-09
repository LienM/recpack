# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Sparse Linear Method

Contains the SLIM algorithm
"""
import numpy as np
import scipy.sparse

from sklearn.linear_model import SGDRegressor

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm
from recpack.matrix import Matrix, to_csr_matrix


class SLIM(ItemSimilarityMatrixAlgorithm):
    """Implementation of the SLIM model.

    SLIM Model described in Ning, Xia, and George Karypis.
    "Slim: Sparse linear methods for top-n recommender systems."
    2011 IEEE 11th International Conference on Data Mining. IEEE, 2011

    Code loosely based on https://github.com/Mendeley/mrec

    :param l1_reg: l1 regularization coefficient, defaults to 0.0005
    :type l1_reg: float, optional
    :param l2_reg: l2 regularization coefficient, defaults to 0.00005
    :type l2_reg: float, optional
    :param fit_intercept: Whether the intercept should be estimated
        or not during gradient descent.
        If False, the data is assumed to be already centered., defaults to True
    :type fit_intercept: bool, optional
    :param ignore_neg_weights: Remove negative weights after training
        to increase speed of predict, defaults to True
    :type ignore_neg_weights: bool, optional
    """

    def __init__(self, l1_reg=0.0005, l2_reg=0.00005, fit_intercept=True, ignore_neg_weights=True):

        super().__init__()

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        # Translate regression parameters into the expected sgd parameters
        self.alpha = self.l1_reg + self.l2_reg
        self.l1_ratio = self.l1_reg / self.alpha
        self.fit_intercept = fit_intercept
        self.ignore_neg_weights = ignore_neg_weights

        # Construct internal model
        self.model = SGDRegressor(
            penalty="elasticnet",
            fit_intercept=fit_intercept,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
        )

    def _compute_similarities(self, work_matrix, item):
        new_matrix = work_matrix.tocoo()
        target = new_matrix.getcol(item)
        data_indices = np.where(new_matrix.col == item)[0]
        new_matrix.data[data_indices] = 0
        self.model.fit(new_matrix, target.toarray().ravel())

        w = self.model.coef_
        if self.ignore_neg_weights:
            w[w < 0] = 0
        return w

    def _fit(self, X: Matrix):
        """Fit a similarity matrix based on data X.

        X is an m x n binary matrix of user item interactions.
        Where m is the number of users, and n the number of items.
        """
        X = to_csr_matrix(X, binary=True)

        # Prep sparse representation inputs
        data = []
        row = []
        col = []
        # Loop over all items
        for j in range(X.shape[1]):
            # Compute the contribution values of all other items for the item j
            # using linear regression
            w = self._compute_similarities(X, j)
            # Update sparse repr. inputs.
            # w[i,j] = the contribution of item i to predicting item j
            for i in w.nonzero()[0]:
                data.append(w[i])
                row.append(i)
                col.append(j)

        # Construct similarity matrix.
        # Shape is determined by 2nd dimension of the shape of input matrix X
        self.similarity_matrix_ = scipy.sparse.csr_matrix((data, (row, col)), shape=(X.shape[1], X.shape[1]))
