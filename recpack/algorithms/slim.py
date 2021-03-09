"""Sparse Linear Method

Contains the SLIM algorithm
"""
import numpy as np
import scipy.sparse

from sklearn.linear_model import SGDRegressor

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class SLIM(ItemSimilarityMatrixAlgorithm):
    """Implementation of the SLIM model.
    loosely based on https://github.com/Mendeley/mrec
    """

    def __init__(
        self,
        l1_reg=0.0005,
        l2_reg=0.00005,
        fit_intercept=True,
        ignore_neg_weights=True,
        model="sgd",
    ):
        super().__init__()

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        # Translate regression parameters into the expected sgd parameters
        self.alpha = self.l1_reg + self.l2_reg
        self.l1_ratio = self.l1_reg / self.alpha
        self.fit_intercept = fit_intercept
        self.ignore_neg_weights = ignore_neg_weights

        # Construct internal model
        # ALLOWED MODELS:
        ALLOWED_MODELS = ["sgd"]

        if model == "sgd":
            self.model = SGDRegressor(
                penalty="elasticnet",
                fit_intercept=fit_intercept,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
            )

        else:
            raise NotImplementedError(
                f"{model} is not yet implemented, "
                f"please use one of {ALLOWED_MODELS}"
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
        self.similarity_matrix_ = scipy.sparse.csr_matrix(
            (data, (row, col)), shape=(X.shape[1], X.shape[1])
        )
