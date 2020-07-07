import scipy.sparse
import numpy

from recpack.algorithms.user_item_interactions_algorithms.linear_algorithms import (
    EASE
)
from recpack.algorithms.user_item_interactions_algorithms.ho_ease import (
    HOEASE,
    ItemsetTransform
)


def test_ease():
    """
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [1 0 1],
            [1 0 1],
            [1 1 1],
            [1 0 1]
        ]

    The algorithm should learn that item 0 is a near-perfect predictor for
    item 2, especially when almost no regularization is applied (low value for l2)
    """
    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))

    algo = EASE(l2=0.03)

    algo.fit(data)

    # Make sure the predictions "make sense"
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result[2, 0], 1, decimal=1)
    numpy.testing.assert_almost_equal(result[0, 2], 1, decimal=1)


def test_hoease_does_ease():
    """
    If we set amt_itemsets = 0, HOEASE should be identical to regular EASE.
    ---
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [1 0 1],
            [1 0 1],
            [1 1 1],
            [1 0 1]
        ]

    The algorithm should learn that item 0 is a near-perfect predictor for
    item 2, especially when almost no regularization is applied (low value for l2)
    """
    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))

    transform = ItemsetTransform(min_freq=0.5, amt_itemsets=0)
    X = transform.fit_transform(data)
    y = data
    algo = HOEASE(l2=0.03)

    algo.fit(X, y, itemset_transform=transform)

    # Make sure the predictions "make sense"
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result[2, 0], 1, decimal=1)
    numpy.testing.assert_almost_equal(result[0, 2], 1, decimal=1)


def test_hoease():
    """
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [1 0 1],
            [1 0 1],
            [1 1 1],
            [1 0 1]
        ]

    The algorithm should learn that item 0 is a near-perfect predictor for
    item 2, especially when almost no regularization is applied (low value for l2)
    """
    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))

    transform = ItemsetTransform(min_freq=0.1, amt_itemsets=1)
    X = transform.fit_transform(data)
    y = data
    algo = HOEASE(l2=0.03)

    algo.fit(X, y, itemset_transform=transform)

    # Make sure the predictions "make sense"
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    _in = transform.transform(_in)
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result[2, 0], 1, decimal=1)
    numpy.testing.assert_almost_equal(result[0, 2], 1, decimal=1)

    assert len(transform.itemsets) == 1
