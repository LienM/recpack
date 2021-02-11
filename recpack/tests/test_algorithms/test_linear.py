import pytest
import scipy.sparse
import numpy

from recpack.algorithms.similarity.linear import EASE


@pytest.fixture()
def data():
    """
    The idea here is to create a matrix that should be super "easy"
    for the algorithm to complete:

        [
            [1 0 1],
            [1 0 1],
            [1 1 1],
            [1 0 1]
        ]

    """

    values = [1] * 9
    users = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 2, 0, 2, 0, 1, 2, 0, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 3))
    return data


def test_ease(data):
    """
    The algorithm should learn that item 0 is a near-perfect predictor for
    item 2, especially when almost no regularization is applied (low value for l2)
    """
    algo = EASE(l2=0.03)

    algo.fit(data)

    # Make sure the predictions "make sense"
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result[2, 0], 1, decimal=1)
    numpy.testing.assert_almost_equal(result[0, 2], 1, decimal=1)


def test_prune(data):
    def density(X):
        n = X.shape[0] * X.shape[1]
        return X.nnz / n

    TARGET_DENSITY = 2 / 9

    algo = EASE(l2=0.03, density=TARGET_DENSITY)
    algo.similarity_matrix_ = scipy.sparse.csr_matrix(
        ([0.9, 0.8, 0.7, 0.2, -0.9, -0.6], ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]))
    )

    assert density(algo.similarity_matrix_) > TARGET_DENSITY
    algo.prune()

    numpy.testing.assert_almost_equal(density(algo.similarity_matrix_), TARGET_DENSITY)
