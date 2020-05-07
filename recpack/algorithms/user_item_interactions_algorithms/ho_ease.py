import numpy as np
import scipy.sparse
from tqdm import tqdm

from recpack.algorithms.user_item_interactions_algorithms.linear_algorithms import EASE

from sklearn.utils.validation import check_is_fitted

from recpack.utils.parfor import parfor
from recpack.utils.monitor import Monitor
from recpack.utils import miner


class HOEASE(EASE):
    def __init__(self, l2=500, minfreq=0.1, amt_itemsets=300):
        super().__init__(l2)
        self.minfreq = minfreq
        self.amt_itemsets = amt_itemsets

    def fit(self, Y, w=None):
        """ Compute the closed form solution of HO-EASE. """
        if w:
            raise RuntimeError("Weights are not supported in HO-EASE")

        monitor = Monitor("HO-EASE")

        Y = scipy.sparse.csr_matrix(Y)

        monitor.update("Compute Itemsets")
        itemsets = compute_itemsets(Y, self.minfreq * Y.shape[1], self.amt_itemsets)

        monitor.update("Add Itemsets")
        itemsetIndex = Y.shape[1]
        X = extend_with_itemsets(Y, itemsets)

        monitor.update("Calculate G")
        G = X.T @ X + self.l2 * np.identity(X.shape[1])

        monitor.update("Invert G")
        P = np.linalg.inv(G)
        del G       # free memory

        monitor.update("Calculate  Brr")
        B_rr = P @ (X.T @ Y).todense()

        # calculate lagrangian multipliers
        monitor.update("Calculate Lagr. Mult. (prepr.)")

        # first do one pass over itemsets to find indices
        Item_indices = {j: [j] for j in range(B_rr.shape[1])}
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

            LM[S, j] = np.squeeze(LM_sj)

        monitor.update("Calculate  B")
        B = B_rr - P @ LM

        self.B_ = scipy.sparse.csr_matrix(B)
        self.itemsets_ = itemsets

        return self

    def predict(self, Y):
        check_is_fitted(self)

        Y = scipy.sparse.csr_matrix(Y)
        X = extend_with_itemsets(Y, self.itemsets_)
        return super().predict(X)

    @property
    def name(self):
        return f"HO-EASE_lambda_{self.l2}"


def extend_with_itemsets(Y, itemsets):
    """ Extends a user-item interaction matrix with itemset interactions. Returns augmented X. """

    if not itemsets:
        return Y

    # Note: could use tidlist to add entries in sparse matrix
    # should be faster than recalculating indices here, but less uniform

    @parfor(itemsets)
    def extensions(itemset):
        itemset = list(itemset)
        vector = Y[:, itemset[0]]
        for item in itemset[1:]:
            vector = Y[:, item].multiply(vector)

        return vector

    X_ext = scipy.sparse.hstack((Y, *extensions), format="csr")

    return X_ext


def compute_itemsets(Y, minsup, amount):
    return miner.calculate_itemsets(Y, minsup=minsup, amount=amount)


