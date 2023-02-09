# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
import time

import numpy as np
import scipy.sparse

from recpack.algorithms import EASE
from recpack.matrix import Matrix, to_csr_matrix

logger = logging.getLogger("recpack")


class EASE_XY(EASE):
    """Variation of EASE, reconstructing a second matrix given during training.

    Instead of autoencoding, trying to reconstruct the training matrix,
    training will try to construct the second matrix y, given the model and matrix X.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import EASE_XY

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))
        y = csr_matrix(np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]]))

        algo = EASE_XY()
        # Fit algorithm
        # Uses interactions in X to predict interactions in y
        algo.fit(X, y)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param l2: regularization parameter to avoid overfitting, defaults to 1e3
    :type l2: float, optional
    :param alpha: parameter to punish popular items.
        Each similarity score between items i and j is divided by count(j)**alpha.
        Defaults to 0
    :type alpha: int, optional
    :param density: Parameter to reduce density of the output matrix,
        significantly speeds up and reduces memory footprint of prediction with a
        little loss of accuracy.
        Does not impact memory consumption of training.
        Defaults to None
    :type density: float, optional

    """

    def fit(self, X: Matrix, y: Matrix) -> "EASE_XY":
        """Fit the model, so it can predict interactions in matrix y, given matrix X

        :param X: Training data
        :type X: Matrix
        :param y: Matrix to predict
        :type y: Matrix, optional
        :return: self
        :rtype: EASE_XY
        """
        start = time.time()
        X, y = to_csr_matrix((X, y), binary=True)

        XTX = X.T @ X
        G = XTX + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(np.diag(B_rr) / np.diag(P))
        B = B_rr - P @ D

        if self.alpha != 0:
            w = 1 / np.diag(XTX.toarray()) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        if self.density:
            self._prune()

        self._check_fit_complete()
        end = time.time()
        logger.info(f"fitting {self.name} complete - Took {start - end :.3}s")

        return self
