from typing import Union

import numpy as np
import scipy.sparse
import logging

from recpack.algorithms.algorithm_base import Algorithm


def get_topK(X_pred: scipy.sparse.csr_matrix, K: int) -> scipy.sparse.csr_matrix : 
    # Get nonzero users
    nonzero_users = list(set(X_pred.nonzero()[0]))
    X = X_pred[nonzero_users, :].toarray()

    items = np.argpartition(X, -K)[:, -K:]

    U, I, V = [], [], []

    for ix, user in enumerate(nonzero_users):
        U.extend([user] * K)
        I.extend(items[ix])
        V.extend(X[ix, items[ix]])

    X_pred_top_K = scipy.sparse.csr_matrix(
        (V, (U, I)), dtype=X_pred.dtype, shape=X_pred.shape
    )

    return X_pred_top_K


class ProductLabeler:
    def __init__(self):
        """
        ProductLabeler labels the purchase history with durable or consumable.
        """
        self.is_fit = False
        self.labels = None

    def fit(self, labels):
        """
        Fit the ProductLabeler so that if an item is both durable
        and it occurs in the user's purchase history,
        it is labeled 1, otherwise zero.
        """
        self.labels = labels
        self.is_fit = True

    def get_durable(self, X):
        """
        Return only the part of X that is durable
        and occured in the user's purchase history.
        """
        return X.multiply(self.labels)

    def get_consumable(self, X):
        """
        Return only the part of X that is not both durable and occured in the user's purchase history.
        """
        durable = X.multiply(self.labels)

        new_X = X.copy()

        nonzero_users = list(set(X.nonzero()[0]))

        new_X[nonzero_users, :] = X[nonzero_users, :] - durable[nonzero_users, :]

        return new_X


class PurchaseHistoryDurableFilter:
    def __init__(self):
        """
        ProductLabeler labels the purchase history with durable or consumable.
        """
        self.is_fit = False
        self.user_labels = None

    def fit(self, labels, purchases):
        """
        Fit the ProductLabeler so that if an item is both durable
        and it occurs in the user's purchase history,
        it is labeled 1, otherwise zero.
        """
        self.user_labels = purchases.multiply(labels)
        self.is_fit = True

    def get_durable(self, X):
        """
        Return only the part of X that is durable
        and occured in the user's purchase history.
        """
        return X.multiply(self.user_labels)

    def get_consumable(self, X):
        """
        Return only the part of X that is not both durable and occured in the user's purchase history.
        """
        durable = X.multiply(self.user_labels)

        new_X = X.copy()

        nonzero_users = list(set(X.nonzero()[0]))

        new_X[nonzero_users, :] = X[nonzero_users, :] - durable[nonzero_users, :]

        return new_X


class RetailAlgorithm(Algorithm):
    def __init__(self):
        pass

    def fit(self, X):
        """
        Fit a model using purchases and optionally pageviews information.

        :param purchases: [description]
        :type purchases: [type]
        :param pageviews: [description], defaults to None
        :type pageviews: [type], optional
        """
        pass

    def fit_classifier(self, labels, purchases):
        pass

    def predict(self, X, user_ids=None):
        pass


class FilterDurableGoods(RetailAlgorithm):
    def __init__(self, rec_algo, goods_classifier):
        self.rec_algo = rec_algo
        self.goods_classifier = goods_classifier

    def fit(self, X):
        """
        Fit the underlying recommendation algorithm.
        X can be pageviews or purchases data, depending on the evaluation setting.

        :param X: [description]
        :type X: [type]
        """
        if not self.goods_classifier.is_fit:
            raise RuntimeError("Goods Classifier should have been fit")

        self.rec_algo.fit(X)

    def fit_classifier(self, labels, purchases):
        self.goods_classifier.fit(labels, purchases)

    def predict(self, X, user_ids=None):
        consumable_X = self.goods_classifier.get_consumable(X)

        return self.rec_algo.predict(consumable_X)

    # @property
    # def name(self):
    #     return f"{self.rec_algo.name}_filter_durable"


class DiscountDurableGoods(RetailAlgorithm):
    def __init__(
        self, rec_algo, goods_classifier, discount_value=1 / 3, K=10):
        self.rec_algo = rec_algo
        self.goods_classifier = goods_classifier
        self.discount_value = discount_value
        self.K = K

    def fit_classifier(self, labels, purchases):
        self.goods_classifier.fit(labels, purchases)

    def fit(self, X):
        """
        Fit the underlying recommendation algorithm.
        X can be pageviews or purchases data, depending on the evaluation setting.

        :param X: [description]
        :type X: [type]
        """
        if not self.goods_classifier.is_fit:
            raise RuntimeError("Goods Classifier should have been fit up front.")

        self.rec_algo.fit(X)

    def predict(self, X, user_ids=None):

        consumable_X = self.goods_classifier.get_consumable(X)
        durable_X = self.goods_classifier.get_durable(X)

        durable_recos = self.rec_algo.predict(durable_X)

        topK_durable_recos = self.get_topK(durable_recos)

        return self.rec_algo.predict(
            consumable_X
        ) - self.discount_value * topK_durable_recos

    def get_topK(self, X_pred: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        return get_topK(X_pred, self.K)

    # @property
    # def name(self):
    #     return f"{self.rec_algo.name}_discount_durable"


class DiscountDurableNeighboursOfDurableItems(DiscountDurableGoods):
    """
    We will still use the durable items for prediction, but their durable neighbours are discounted.

    The goal is to reduce the space spent on alternatives for already consumed durable items,
    without removing accessories or other related items that are still relevant for the user.

    :param rec_algo: The recommendation algorithm performing recommendations
    :type rec_algo: Algorithm

    :param goods_classifier: The goods classifier who will classify an item as durable or not
    :type goods_classifier: GlobalProductLabeler

    :param user_goods_classifier: A goods classifier which will classify an item as durable
                                  or not based on user interactions with that item
    :type user_goods_classifier: ProductLabeler


    :param discount_value: The amount with which durable neighbours will be discounted. Default = 1, 
                            so that the contribution of the durable items NN scores is removed.
                            In order to remove the items use a discount_value of 2 or higher
    :type discount_value: float

    :param K: The top K neighbours of each item will be considered for discounting.
    :type K: int

    """
    def __init__(self, rec_algo, goods_classifier, user_goods_classifier, discount_value=1, K=10):
        self.rec_algo = rec_algo
        self.goods_classifier = goods_classifier
        self.user_goods_classifier = user_goods_classifier
        self.discount_value = discount_value
        self.K = K

    def fit_classifier(self, labels, purchases):
        self.goods_classifier.fit(labels)
        self.user_goods_classifier.fit(labels, purchases)

    def fit(self, X):
        """
        Fit the underlying recommendation algorithm.
        X can be pageviews or purchases data, depending on the evaluation setting.

        :param X: [description]
        :type X: [type]
        """
        if not self.goods_classifier.is_fit:
            raise RuntimeError("Goods Classifier should have been fit up front.")

        if not self.user_goods_classifier.is_fit:
            raise RuntimeError("User Goods Classifier should have been fit up front.")

        self.rec_algo.fit(X)
 
    def predict(self, X, user_ids=None):
        """
        Predict scores given X the user interaction matrix.

        score computed = prediction(X) - discount * durable_items(topK(prediction(user_durable_items(X))))

        :param X: The user interaction matrix
        :type X: scipy.sparse.csr_matrix
        """

        # Get the items the user has either purchased but are consumable or items that are durable,
        # but the user has not yet purchased
        durable_X = self.user_goods_classifier.get_durable(X)
        
        reco_scores = self.rec_algo.predict(X)

        # For each nonzero item in durable_X get the score for that item, and discount it from the user's that have seen that item.
        for i in set(durable_X.nonzero()[1]):
            m = scipy.sparse.csr_matrix(([1], ([0], [i])), shape=(1, X.shape[1]))
            # Discount the items the durable neighbours
            pred = self.rec_algo.predict(durable_X.multiply(m))

            reco_scores -= self.discount_value * self.goods_classifier.get_durable(self.get_topK(pred))

        return reco_scores

    # @property
    # def name(self):
    #     return f"{self.rec_algo.name}_discount_durable_neighbours_{self.discount_value}_@_{self.K}"


class DiscountAlternativesOfDurableItems(DiscountDurableGoods):
    """
    This approach is very similar to DiscountDurableNeighboursOfDurableItems,
    Where for that approach we used the recommender algorithm to also do the discounting, 
    We will now use a special second algorithm to compute alternatives for the durable items,
    those will then get discounted.

    :param rec_algo: The recommendation algorithm performing recommendations
    :type rec_algo: Algorithm

    :param alternatives_algo: The algorithm predicting alternatives for items.
    :type alternatives_algo: Algorithm

    :param goods_classifier: The goods classifier who will classify an item as durable or not
    :type goods_classifier: GlobalProductLabeler

    :param user_goods_classifier: A goods classifier which will classify an item as durable
                                  or not based on user interactions with that item
    :type user_goods_classifier: ProductLabeler

    :param discount_value: The amount with which durable neighbours will be discounted. Default = 1, 
                            so that the contribution of the durable items NN scores is removed.
                            In order to remove the items use a discount_value of 2 or higher
    :type discount_value: float

    :param K: The top K neighbours of each item will be considered for discounting.
    :type K: int

    """

    def __init__(self, rec_algo, alternatives_algo, goods_classifier, user_goods_classifier, discount_value=1, K=10):
        self.rec_algo = rec_algo
        self.alternatives_algo = alternatives_algo
        self.goods_classifier = goods_classifier
        self.user_goods_classifier = user_goods_classifier
        self.discount_value = discount_value
        self.K = K

    def fit_classifier(self, labels, X):
        self.goods_classifier.fit(labels)
        self.user_goods_classifier.fit(labels, X)

    def fit(self, X):
        """
        Fit the underlying recommendation algorithm.
        X can be pageviews or purchases data, depending on the evaluation setting.

        :param X: [description]
        :type X: [type]
        """
        if not self.goods_classifier.is_fit:
            raise RuntimeError("Goods Classifier should have been fit up front.")

        if not self.user_goods_classifier.is_fit:
            raise RuntimeError("User Goods Classifier should have been fit up front.")

        self.rec_algo.fit(X)
        self.alternatives_algo.fit(X)

    def predict(self, X, user_ids=None):
        """
        Predict scores given X the user interaction matrix.

        score computed = prediction(X) - discount topK(alternatives(user_durable_items(X)))

        :param X: The user interaction matrix
        :type X: scipy.sparse.csr_matrix
        """

        # Get the items the user has either purchased but are consumable or items that are durable,
        # but the user has not yet purchased
        durable_X = self.user_goods_classifier.get_durable(X)
        
        reco_scores = self.rec_algo.predict(X)

        # For each nonzero item in durable_X get the score for that item, and discount it from the user's that have seen that item.
        for i in set(durable_X.nonzero()[1]):
            m = scipy.sparse.csr_matrix(([1], ([0], [i])), shape=(1, X.shape[1]))
            # Discount the items the durable neighbours
            pred = self.alternatives_algo.predict(durable_X.multiply(m))

            reco_scores -= self.discount_value * self.goods_classifier.get_durable(self.get_topK(pred))

        return reco_scores

    # @property
    # def name(self):
    #     return f"{self.rec_algo.name}_discount_{self.alternatives_algo.name}"
