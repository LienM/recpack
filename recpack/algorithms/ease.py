# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging

import numpy as np
import scipy.sparse

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm
from recpack.matrix import Matrix, to_csr_matrix

logger = logging.getLogger("recpack")


class EASE(ItemSimilarityMatrixAlgorithm):
    """Implementation of the EASEr algorithm.

    Implementation of the Embarrassingly Shallow Autoencoder as presented in
    Steck, Harald. "Embarrassingly shallow autoencoders for sparse data."
    The World Wide Web Conference. 2019.

    The algorithm essentially computes a high-dimensional linear autoencoder.
    Constructs a similarity matrix :math:`B` with 0 diagonal which minimises:

    .. math::

        ||X \\cdot \\text{diagMat}(w) - X \\cdot B||_F^2 + \\lambda \\cdot ||B||_F^2

    where :math:`w` is an array with importance weights per item: :math:`w_i = \\frac{1}{pop(i)^\\alpha}`

    Thanks to a closed form solution this algorithm has a significant speed up
    compared to the SLIM algorithm on which it is based.

    .. warning::

        Memory consumption scales worse than quadratically in the amount of items.
        So check the size of the input matrix before using this algorithm.

    :param l2: Regularization parameter to avoid overfitting, defaults to `1e3`.
    :type l2: float, optional
    :param alpha: Parameter to punish popular items.
        Each similarity score between items i and j is divided by count(j)**alpha.
        Defaults to 0
    :type alpha: int, optional
    :param density: Parameter to reduce density of the output matrix,
        significantly speeds up and reduces memory footprint of prediction with only a
        small loss of accuracy.
        Does not impact memory consumption of training.
        Defaults to None
    :type density: float, optional
    """

    def __init__(self, l2=1e3, alpha=0, density=None):
        super().__init__()
        self.l2 = l2
        self.alpha = alpha  # alpha exponent for filtering popularity bias
        self.density = density

    def _fit(self, X: Matrix):
        """Compute the closed form solution,
        optionally rescalled to counter popularity bias (see param alpha).

        Dense linear model algorithm with closed-form solution
        Embarrassingly shallow auto-encoder from Steck @ WWW 2019
        https://arxiv.org/pdf/1905.03375.pdf
        Dense version in Steck et al. @ WSDM 2020
        http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf
        Eq. 21: B = I - P · diagMat(1 ⊘ diag(P)
        More info on the solution for rescaling targets in section 4.2 of
        Collaborative Filtering via High-Dimensional Regression from Steck
        https://arxiv.org/pdf/1904.13033.pdf
        Eq. 14 B_scaled = B * diagM(w)
        """
        X = to_csr_matrix(X, binary=True)

        # Compute P
        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(XTX + self.l2 * np.identity((X.shape[1]), dtype=np.float32))

        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = 0.0

        if self.alpha != 0:
            w = 1 / np.diag(XTX) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        if self.density:
            self._prune()

    def _prune(self):
        """Prune the similarity matrix

        Steck et al. state that we can increase the sparsity in matrix B
        without significant impact on quality.
        """

        K = min(
            int(self.density * np.product(self.similarity_matrix_.shape)),
            self.similarity_matrix_.nnz,
        )
        self.similarity_matrix_.data[np.argpartition(abs(self.similarity_matrix_.data), -K)[0:-K]] = 0
        self.similarity_matrix_.eliminate_zeros()
