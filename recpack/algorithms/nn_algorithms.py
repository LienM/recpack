from collections import Counter, defaultdict
import math

from scipy.sparse import diags

from .algorithm_base import Algorithm


class ItemKNN(Algorithm):

    def __init__(self, K):
        self.K = K
        self.item_co_occurrences = None

    def fit(self, X):

        co_mat = X.T @ X
        # TODO This division could also be done on an item-basis when predicting. Not sure which is better.
        A = diags(1 / co_mat.diagonal())
        # This has all item-co-occurences. Now we should probably set N-K to zero
        self.item_co_occurences = A @ co_mat

    def predict(self, X):
        # TODO Implement. Wasn't sure how ItemKNN choses? 
        pass


class SharedAccount(ItemKNN):

    def __init__(self, K):
        super().__init__(K)

    def predict(self, X):
        pass