# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms import Popularity, Random


def test_random(X_in):
    seed = 42
    K = 2
    algo = Random(K=K, seed=seed)
    algo.fit(X_in)

    X_pred = algo.predict(X_in)

    assert X_pred.nnz == len(set(X_pred.nonzero()[0])) * K


def test_random_use_only_interacted_items(X_in):
    algo1 = Random(K=2, use_only_interacted_items=True)
    algo2 = Random(K=2, use_only_interacted_items=False)
    algo1.fit(X_in)
    algo2.fit(X_in)
    assert len(algo1.items_) < len(algo2.items_)


def test_random_K_is_None(X_in):
    algo1 = Random(K=None, use_only_interacted_items=True)
    algo2 = Random(K=None, use_only_interacted_items=False)
    algo1.fit(X_in)
    algo2.fit(X_in)
    assert len(algo1.items_) < len(algo2.items_)

    X_pred = algo1.predict(X_in)

    assert len(set(X_pred.nonzero()[1]).difference(set(X_in.nonzero()[1]))) == 0


def test_popularity():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = csr_matrix((values, (user_i, item_i)))
    algo = Popularity(K=20)

    algo.fit(train_data)

    _in = csr_matrix(([1, 1], ([0, 1], [1, 1])), shape=(5, 5))
    prediction = algo.predict(_in)

    # All users in _in get the same recommendations
    np.testing.assert_almost_equal(prediction[0, :].toarray(), prediction[1, :].toarray())
    # The most popular item is ranked highest
    assert prediction[0, 4] > prediction[0, 3]
    # Users who were not in _in do not receive any recommendations
    assert prediction[2, :].nnz == 0


def test_popularity_K_larger_than_num_items():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = csr_matrix((values, (user_i, item_i)))
    algo = Popularity(K=20)
    with warnings.catch_warnings(record=True) as w:
        algo.fit(train_data)
        assert len(w) > 0
        assert "K is larger than the number of items." in str(w[-1].message)
