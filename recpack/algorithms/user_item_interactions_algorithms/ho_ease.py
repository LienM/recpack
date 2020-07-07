import numpy as np
import scipy.sparse
from tqdm import tqdm

from recpack.algorithms.user_item_interactions_algorithms.linear_algorithms import EASE

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from recpack.utils.parfor import parfor
from recpack.utils.monitor import Monitor
from recpack.utils import miner


class HOEASE(EASE):
    def __init__(self, l2=500):
        super().__init__(l2)

    def fit(self, X: scipy.sparse.csr_matrix, y: scipy.sparse.csr_matrix=None, itemset_transform=None):
        """
        Compute the solution for HO-EASE in closed-form.
        We follow the sklearn notation where X is the training data.
        In the derivation Y corresponds to X and X corresponds to X_ext.

        :param X: User-item interaction matrix, expected binary.
        :type X: scipy.sparse.csr_matrix
        :param w: Weights to apply to the items, defaults to None
        :type w: [type], optional
        :raises RuntimeError: [description]
        :return: self
        :rtype: type
        """
        if y is None:
            raise ValueError("Need to provide interaction matrix (without extensions) as y in HO-EASE")

        if itemset_transform is None:
            raise ValueError("Need itemsets to calculate correct indices")

        monitor = Monitor("HO-EASE")

        monitor.update("Calculate G")
        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        monitor.update("Invert G")
        P = np.linalg.inv(G)
        del G       # free memory

        monitor.update("Calculate  Brr")
        B_rr = P @ (X.T @ y).todense()

        # calculate lagrangian multipliers
        monitor.update("Calculate Lagr. Mult. (prepr.)")

        # first do one pass over itemsets to find indices
        itemset_index = y.shape[1]
        itemsets = itemset_transform.itemsets

        item_indices = {j: [j] for j in range(B_rr.shape[1])}
        for i, itemset in enumerate(itemsets):
            for j in itemset:
                item_indices[j].append(i + itemset_index)

        monitor.update("Calculate Lagr. Mult. (invert)")
        # then calculate multipliers
        LM = np.zeros(B_rr.shape)
        for j in tqdm(range(LM.shape[1])):
            S = item_indices[j]

            P_ss = P[np.ix_(S, S)]
            B_rr_sj = B_rr[S, j]

            # pinv to account for numerical errors
            LM_sj = np.linalg.pinv(P_ss) @ B_rr_sj
            # TODO: could use linalg.lstsq or solve

            LM[S, j] = np.squeeze(LM_sj)

        monitor.update("Calculate  B")
        B = B_rr - P @ LM

        self.B_ = scipy.sparse.csr_matrix(B)

        return self


class ItemsetTransform(BaseEstimator, TransformerMixin):
    """ Extends a user-item interaction matrix with itemset interactions. Returns augmented X. """

    def __init__(self, min_freq=0.1, amt_itemsets=300):
        super().__init__()
        self.min_freq = min_freq
        self.amt_itemsets = amt_itemsets

    @property
    def itemsets(self):
        """ lazy loading of itemsets """
        return self.itemsets_

    def fit(self, X, y=None):
        print("Calculating itemsets")
        self.itemsets_ = miner.calculate_itemsets(X, minsup=self.min_freq * X.shape[1], amount=self.amt_itemsets)
        return self

    def transform(self, X):
        check_is_fitted(self)
        print("Extending features with itemsets")

        assert scipy.sparse.issparse(X), "Sparse matrix required"

        # Note: could use tidlist to add entries in sparse matrix
        # should be faster than recalculating indices here, but less uniform

        @parfor(self.itemsets_)
        def extensions(itemset):
            itemset = list(itemset)
            vector = X[:, itemset[0]]
            for item in itemset[1:]:
                vector = X[:, item].multiply(vector)

            return vector

        X_ext = scipy.sparse.hstack((X, *extensions), format="csr")

        return X_ext
