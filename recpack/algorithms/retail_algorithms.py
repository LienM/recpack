from typing import Union

import numpy as np
import scipy.sparse
import logging

from recpack.algorithms.algorithm_base import Algorithm


class ProductLabeler:
    def __init__(self):
        """
        ProductLabeler labels the purchase history with durable or consumable.
        """
        self.is_fit = False
        self.user_discount = None

    def fit(self, labels, purchases):
        """
        Fit the ProductLabeler so that if an item is both durable
        and it occurs in the user's purchase history,
        it is labeled 1, otherwise zero.
        """
        self.user_discount = purchases.multiply(labels)
        self.is_fit = True

    def get_durable(self, X):
        """
        Return only the part of X that is durable
        and occured in the user's purchase history.
        """
        return X.multiply(self.user_discount)

    def get_consumable(self, X):
        """
        Return only the part of X that is not both durable and occured in the user's purchase history.
        """
        durable = X.multiply(self.user_discount)

        new_X = X.copy()

        nonzero_users = list(set(X.nonzero()[0]))

        new_X[nonzero_users, :] = X[nonzero_users, :] - durable[nonzero_users, :]

        return new_X


class GlobalProductLabeler:
    def __init__(self):
        """
        ProductLabeler labels the purchase history with durable or consumable.
        """
        self.is_fit = False
        self.user_discount = None

    def fit(self, labels):
        """
        Fit the ProductLabeler so that if an item is both durable
        and it occurs in the user's purchase history,
        it is labeled 1, otherwise zero.
        """
        self.user_discount = labels
        self.is_fit = True

    def get_durable(self, X):
        """
        Return only the part of X that is durable
        and occured in the user's purchase history.
        """
        return X.multiply(self.user_discount)

    def get_consumable(self, X):
        """
        Return only the part of X that is not both durable and occured in the user's purchase history.
        """
        durable = X.multiply(self.user_discount)

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

    def predict(self, X):
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

    def predict(self, X):
        consumable_X = self.goods_classifier.get_consumable(X)

        return self.rec_algo.predict(consumable_X)

    @property
    def name(self):
        return f"{self.rec_algo.name}_filter_durable"


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

    def predict(self, X):

        consumable_X = self.goods_classifier.get_consumable(X)
        durable_X = self.goods_classifier.get_durable(X)

        durable_recos = self.rec_algo.predict(durable_X)

        topK_durable_recos = self.get_topK(durable_recos)

        return self.rec_algo.predict(
            consumable_X
        ) - self.discount_value * topK_durable_recos

    def get_topK(self, X_pred: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        # Get nonzero users
        nonzero_users = list(set(X_pred.nonzero()[0]))
        X = X_pred[nonzero_users, :].toarray()

        items = np.argpartition(X, -self.K)[:, -self.K:]

        U, I, V = [], [], []

        for ix, user in enumerate(nonzero_users):
            U.extend([user] * self.K)
            I.extend(items[ix])
            V.extend(X[ix, items[ix]])

        X_pred_top_K = scipy.sparse.csr_matrix(
            (V, (U, I)), dtype=X_pred.dtype, shape=X_pred.shape
        )

        return X_pred_top_K

    @property
    def name(self):
        return f"{self.rec_algo.name}_discount_durable_{self.discount_value}"


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
 
    def predict(self, X):
        """
        Predict scores given X the user interaction matrix.

        score computed = prediction(consumable_items(X)) + prediction(user_durable_items(X)) - durable_items(topK(prediction(user_durable_items(X))))

        :param X: The user interaction matrix
        :type X: scipy.sparse.csr_matrix
        """

        # Get the items the user has either purchased but are consumable or items that are durable,
        # but the user has not yet purchased
        consumable_X = self.user_goods_classifier.get_consumable(X)

        durable_X = self.user_goods_classifier.get_durable(X)

        durable_recos = self.rec_algo.predict(durable_X)
        # discount the durable neighbours in the top K
        durable_recos_discounted = durable_recos - self.discount_value * self.goods_classifier.get_durable(self.get_topK(durable_recos))

        print("durable_recos:")
        print(durable_recos.toarray())
        print("durable_recos_discounted")
        print(durable_recos_discounted.toarray())
        print("topk_durable_recos")
        print(self.get_topK(durable_recos).toarray())
        print("durable from topK")
        print(self.goods_classifier.get_durable(self.get_topK(durable_recos)))
        print("durable_items")
        print(self.goods_classifier.get_durable(scipy.sparse.dia_matrix(5).tocsr()))


        return self.rec_algo.predict(
            consumable_X
        ) + durable_recos_discounted

    @property
    def name(self):
        return f"{self.rec_algo.name}_discount_durable_neighbours_{self.discount_value}"