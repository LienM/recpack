import numpy as np
import scipy.sparse

from recpack.algorithms.base import SimilarityMatrixAlgorithm
from recpack.data.matrix import Matrix, to_csr_matrix


class EASE(SimilarityMatrixAlgorithm):
    def __init__(self, l2=1e3, alpha=0, density=None):
        """ l2 norm for regularization and alpha exponent to filter popularity bias. """
        super().__init__()
        self.l2 = l2
        self.alpha = alpha  # alpha exponent for filtering popularity bias
        self.density = density

    def fit(self, X: Matrix):
        """Compute the closed form solution, optionally rescalled to counter popularity bias (see param alpha). """
        # Dense linear model algorithm with closed-form solution
        # Embarrassingly shallow auto-encoder from Steck @ WWW 2019
        # https://arxiv.org/pdf/1905.03375.pdf
        # Dense version in Steck et al. @ WSDM 2020
        # http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf
        # Eq. 21: B = I − P · diagMat(1 ⊘ diag(P)
        # More info on the solution for rescaling targets in section 4.2 of
        # Collaborative Filtering via High-Dimensional Regression from Steck
        # https://arxiv.org/pdf/1904.13033.pdf
        # Eq. 14 B_scaled = B * diagM(w)
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

        self._check_fit_complete()
        return self

    def _prune(self):
        # Prune B (similarity matrix)
        # Steck et al. state that we can increase the sparsity in matrix B without significant impact on quality.

        K = min(
            int(self.density * np.product(self.similarity_matrix_.shape)),
            self.similarity_matrix_.nnz,
        )
        self.similarity_matrix_.data[
            np.argpartition(abs(self.similarity_matrix_.data), -K)[0:-K]
        ] = 0
        self.similarity_matrix_.eliminate_zeros()


class EASE_XY(EASE):
    """ Variation of EASE where we encode Y from X (no autoencoder). """

    def fit(self, X: Matrix, y: Matrix = None):
        if y is None:
            raise RuntimeError(
                "Train regular EASE (with X=Y) using the EASE algorithm, not EASE_XY."
            )
        X, y = to_csr_matrix((X, y), binary=True)

        XTX = X.T @ X
        G = XTX + self.l2 * np.identity(X.shape[1])

        P = np.linalg.inv(G)
        B_rr = P @ (X.T @ y).todense()

        D = np.diag(np.diag(B_rr) / np.diag(P))
        B = B_rr - P @ D

        if self.alpha != 0:
            w = 1 / np.diag(XTX) ** self.alpha
            B = B @ np.diag(w)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

        if self.density:
            self._prune()

        self._check_fit_complete()

        return self
