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
def mat():
    data = {
        TIMESTAMP_IX: [1, 1, 1, 2, 3, 4],
        ITEM_IX: [0, 0, 1, 2, 1, 2],
        USER_IX: [0, 1, 2, 2, 1, 0]
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.mark.parametrize(
    "decay_fn, decay_coeff, expected_similarities, expected_out",
    [
        (
            "linear",
            0.5,
            numpy.array(
                [
                    [0., 2 / 3, 1 / 2],
                    [2 / 3, 0., 5 / 6],
                    [1 / 2, 5 / 6, 0.]
                ]
            ),
            [[2 / 3, 2 / 3, 4 / 3]]
        ),
        (
            "concave",
            0.5,
            numpy.array(
                [
                    [0., 1 / 4, 1 / 8],
                    [1 / 4, 0., 1 / 2],
                    [1 / 8, 1 / 2, 0.]
                ]
            ),
            [[1 / 4, 1 / 4, 5 / 8]]
        ),
        (
            "convex",
            0.5,
            numpy.array(
                [
                    [0., 1 / 2, 0.],
                    [1 / 2, 0., 3 / 4],
                    [0., 3 / 4, 0.]
                ]
            ),
            [[1 / 2, 1 / 2, 3 / 4]]
        ),
    ]
)
def test_time_decay_knn(mat, decay_fn, decay_coeff, expected_similarities, expected_out):

    algo = TimeDecayingNearestNeighbour(K=2, decay_coeff=decay_coeff, decay_fn=decay_fn)

    algo.fit(mat)

    numpy.testing.assert_almost_equal(
        algo.similarity_matrix_.toarray(), expected_similarities
    )

    # Make sure the similarities recommended are the similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    numpy.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    result = algo.predict(_in)
    numpy.testing.assert_almost_equal(result.toarray(), expected_out)
