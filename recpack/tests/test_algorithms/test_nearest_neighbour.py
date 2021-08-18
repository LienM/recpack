import math
import numpy
import pytest
import scipy.sparse

from recpack.algorithms import ItemKNN


@pytest.fixture(scope="function")
def data():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    d = scipy.sparse.csr_matrix((values, (users, items)), shape=(4, 3))

    return d


@pytest.fixture(scope="function")
def data_empty_col():
    values = [1] * 5
    users = [0, 0, 1, 1, 2]
    items = [1, 2, 2, 1, 2]
    d = scipy.sparse.csr_matrix((values, (users, items)))

    return d


def test_item_knn(data):

    algo = ItemKNN(K=2)

    algo.fit(data)

    expected_similarities = numpy.array(
        [
            [0, 0.5, 2 / math.sqrt(6)],
            [0.5, 0, 2 / math.sqrt(6)],
            [2 / math.sqrt(6), 2 / math.sqrt(6), 0],
        ]
    )
    numpy.testing.assert_almost_equal(
        algo.similarity_matrix_.toarray(), expected_similarities
    )

    # Make sure the similarities recommended are the cosine similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(
        ([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    expected_out = [[0.5, 0.5, 4 / math.sqrt(6)]]
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result.toarray(), expected_out)


def test_item_knn_normalize(data):

    algo = ItemKNN(K=2, normalize=True)

    algo.fit(data)

    numpy.testing.assert_array_almost_equal(
        algo.similarity_matrix_.sum(axis=1), 1)


def test_item_knn_empty_col(data_empty_col):
    algo = ItemKNN(K=2)

    algo.fit(data_empty_col)
    expected_similarities = numpy.array(
        [[0.0, 0.0, 0.0], [0.0, 0, 2 /
                           math.sqrt(6)], [0.0, 2 / math.sqrt(6), 0]]
    )
    numpy.testing.assert_almost_equal(
        algo.similarity_matrix_.toarray(), expected_similarities
    )


def test_item_knn_conditional_probability(data):
    algo = ItemKNN(K=2, similarity="conditional_probability")

    algo.fit(data)
    # similarity is computed as count(i^j) / (count(i) + 1)

    # data matrix looks like
    # 0 1 1
    # 1 0 1
    # 1 1 1

    # cooc = XtX
    # 2 1 2
    # 1 2 2
    # 2 2 3

    # fmt: off
    expected_similarities = numpy.array(
        [
            [0, 1 / 2, 2 / 2],
            [1 / 2, 0, 2 / 2],
            [2 / 3, 2 / 3, 0]
        ]
    )
    # fmt: on
    numpy.testing.assert_almost_equal(
        algo.similarity_matrix_.toarray(), expected_similarities
    )


def test_item_knn_conditional_probability(data):
    algo = ItemKNN(K=2, similarity="conditional_probability", pop_discount=1)

    algo.fit(data)
    # similarity is computed as count(i^j) / (count(i) * count(j) ^ pop_discount)

    # data matrix looks like
    # 0 1 1
    # 1 0 1
    # 1 1 1

    # cooc = XtX
    # 2 1 2
    # 1 2 2
    # 2 2 3

    # fmt: off
    expected_similarities = numpy.array(
        [
            [0, 1 / (2 * 2), 2 / (2 * 3)],
            [1 / (2 * 2), 0, 2 / (2 * 3)],
            [2 / (3 * 2), 2 / (3 * 2), 0]
        ]
    )
    # fmt: on
    numpy.testing.assert_almost_equal(
        algo.similarity_matrix_.toarray(), expected_similarities
    )
