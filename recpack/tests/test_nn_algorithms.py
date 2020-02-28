import recpack.algorithms
import scipy.sparse
import math
import numpy


def test_item_knn():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)))

    algo = recpack.algorithms.get_algorithm("itemKNN")(K=2)

    algo.fit(data)

    expected_similarities = numpy.array([
        [0, 0.5, 2/math.sqrt(6)],
        [0.5, 0, 2/math.sqrt(6)],
        [2/math.sqrt(6), 2/math.sqrt(6), 0]
    ])
    numpy.testing.assert_almost_equal(algo.item_cosine_similarities.toarray(), expected_similarities)

    # Make sure the similarities recommended are the cosine similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)
    print(result)
    print(expected_similarities)
    numpy.testing.assert_almost_equal(result, expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    expected_out = [[0.5, 0.5, 4/math.sqrt(6)]]
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result, expected_out)


def test_item_knn_empty_col():
    values = [1] * 5
    users = [0, 0, 1, 1, 2]
    items = [1, 2, 2, 1, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)))
    algo = recpack.algorithms.get_algorithm("itemKNN")(K=2)

    algo.fit(data)
    expected_similarities = numpy.array([
        [0., 0., 0.],
        [0., 0, 2/math.sqrt(6)],
        [0., 2/math.sqrt(6), 0]
    ])
    numpy.testing.assert_almost_equal(algo.item_cosine_similarities.toarray(), expected_similarities)