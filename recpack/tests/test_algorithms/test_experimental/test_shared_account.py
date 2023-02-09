# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.algorithms.experimental.shared_account import DAMIBCover
from recpack.algorithms.nearest_neighbour import ItemKNN
import scipy.sparse
import math
import numpy


def test_item_knn_sa_is_iknn():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(4, 3))

    algo = DAMIBCover(algo=ItemKNN(K=2), p=0)

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
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    expected_out = [[0.5, 0.5, 4 / math.sqrt(6)]]
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result.toarray(), expected_out)


def test_item_knn_sa():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)), shape=(4, 3))

    algo = DAMIBCover(algo=ItemKNN(K=2), p=0.75)

    algo.fit(data)

    expected_similarities = numpy.array(
        [
            [0, 0.5, 2 / math.sqrt(6)],
            [0.5, 0, 2 / math.sqrt(6)],
            [2 / math.sqrt(6), 2 / math.sqrt(6), 0],
        ]
    )

    # Make sure the similarities recommended are the cosine similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    expected_out = [[0.5, 0.5, 4 / math.sqrt(6) / 2 ** 0.75]]
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result.toarray(), expected_out)
