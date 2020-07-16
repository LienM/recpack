import scipy.sparse
import numpy

from recpack.algorithms.user_item_interactions_algorithms.linear_algorithms import (
    EASE
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


def test_ease_add():
    """ Check that adding 2 identical models together correctly doubles the values in the matrix
    """
    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))

    algo = EASE(l2=0.03)
    algo_2 = EASE(l2=0.03)

    algo.fit(data)
    algo_2.fit(data)

    algo.add(algo_2)

    numpy.testing.assert_array_equal(algo.B_.toarray(), algo_2.B_.multiply(2).toarray())

def test_ease_mult():
    """ Check that when multiplying a model, it's B_ get's updated correctly.
    """
    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))

    algo = EASE(l2=0.03)
    algo_2 = EASE(l2=0.03)

    algo.fit(data)
    algo_2.fit(data)

    algo.multiply(2)

    numpy.testing.assert_array_equal(algo.B_.toarray(), algo_2.B_.multiply(2).toarray())
