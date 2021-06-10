from recpack.algorithms.experimental.time_decay_nearest_neighbour import TimeDecayingNearestNeighbour
import scipy.sparse
import pandas as pd
import numpy
import pytest
from recpack.data.matrix import InteractionMatrix


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def mat_no_timestamp():
    data = {
        TIMESTAMP_IX: [0, 0, 0, 0, 0, 0, 0],
        ITEM_IX: [0, 0, 1, 1, 2, 2, 2],
        USER_IX: [1, 2, 0, 2, 0, 1, 2]
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture(scope="function")
def mat():
    data = {
        TIMESTAMP_IX: [0, 1, 0, 1, 0, 1, 2],
        ITEM_IX: [0, 0, 1, 1, 2, 2, 2],
        USER_IX: [1, 2, 0, 2, 0, 1, 2]
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


def test_time_decay_knn(mat):

    algo = TimeDecayingNearestNeighbour(K=2)

    algo.fit(mat)

    expected_similarities = numpy.array(
        [
            [0., 1., 2.],
            [1., 0., 2.],
            [2., 2., 0.]
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
    expected_out = [[1., 1., 4.]]
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result.toarray(), expected_out)


@pytest.mark.parametrize(
    "decay_fn, decay_coeff", [
        ("linear", 0.5),
        ("convex", 0.5),
    ]
)
def test_time_decay_knn_no_timestamps(mat_no_timestamp, decay_fn, decay_coeff):

    algo = TimeDecayingNearestNeighbour(K=2, decay_fn=decay_fn, decay_coeff=decay_coeff)

    algo.fit(mat_no_timestamp)

    expected_similarities = numpy.array(
        [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
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
    expected_out = [[0., 0., 0.]]
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result.toarray(), expected_out)
