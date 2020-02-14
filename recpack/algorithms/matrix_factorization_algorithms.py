import scipy.sparse
import sklearn.decomposition

from .algorithm_base import Algorithm


class MF(Algorithm):
    def __init__(self, K=100):
        self.K = K

        self.similarity_matrix = None

    def fit(self, X):
        pass

    def predict(self, X):
        if self.similarity_matrix is None:
            raise Exception("Fit a model before trying to predict with it.")

        # TODO again the same similarity approach
        scores = X @ self.similarity_matrix
        return scores

    @property
    def name(self):
        return f"MF_K_{self.K}"


class NMF(MF):
    # TODO check params NMF to see which ones are useful.

    def fit(self, X):
        # Using Sklearn NMF implementation. For info and parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = sklearn.decomposition.NMF(n_components=self.K, init='random', random_state=0)

        # Factorization is W * H. Where W contains user latent vectors, and H contains item latent vectors
        _ = model.fit_transform(X)
        H = model.components_

        # Compute an item to item similarity matrix by self multiplying the latent factors.
        self.similarity_matrix = H.T @ H

    @property
    def name(self):
        return f"NMF_K_{self.K}"


class SVD(MF):

    def fit(self, X):
        # TODO use other parameter options?
        model = sklearn.decomposition.TruncatedSVD(n_components=self.K, n_iter=7, random_state=42)
        model.fit(X)

        # Factorization computes U x Sigma x V
        # Item similarity can then be computed as V.T * Sigma * V
        V = model.components_
        sigma = scipy.sparse.diags(model.singular_values_)
        self.similarity_matrix = V.T @ sigma @ V

    @property
    def name(self):
        return f"SVD_K_{self.K}"
