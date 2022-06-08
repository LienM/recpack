from recpack.algorithms.experimental.time_decay_nearest_neighbour import (
    TimeDecayingNearestNeighbour,
)
import scipy.sparse
import pandas as pd
import numpy as np
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
        USER_IX: [0, 1, 2, 2, 1, 0],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture(scope="function")
def mat_no_timestamps():
    data = {
        TIMESTAMP_IX: [0, 0, 0, 0, 0, 0],
        ITEM_IX: [0, 0, 1, 2, 1, 2],
        USER_IX: [0, 1, 2, 2, 1, 0],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.mark.parametrize(
    "decay_fn, decay_coeff, expected_similarities, expected_out",
    [
        (
            "linear",
            0.5,
            np.array([[0.0, 2 / 3, 1 / 2], [2 / 3, 0.0, 5 / 6], [1 / 2, 5 / 6, 0.0]]),
            [[2 / 3, 2 / 3, 4 / 3]],
        ),
        (
            "concave",
            0.5,
            np.array([[0.0, 1 / 4, 1 / 8], [1 / 4, 0.0, 1 / 2], [1 / 8, 1 / 2, 0.0]]),
            [[1 / 4, 1 / 4, 5 / 8]],
        ),
        (
            "convex",
            0.5,
            np.array([[0.0, 1 / 2, 0.0], [1 / 2, 0.0, 3 / 4], [0.0, 3 / 4, 0.0]]),
            [[1 / 2, 1 / 2, 3 / 4]],
        ),
    ],
)
def test_time_decay_knn(mat, decay_fn, decay_coeff, expected_similarities, expected_out):

    algo = TimeDecayingNearestNeighbour(K=2, decay_coeff=decay_coeff, decay_fn=decay_fn, decay_interval=1)

    algo.fit(mat)

    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)

    # Make sure the similarities recommended are the similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    np.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    result = algo.predict(_in)
    np.testing.assert_almost_equal(result.toarray(), expected_out)


@pytest.mark.parametrize(
    "decay_fn, decay_coeff, expected_similarities, expected_out",
    [
        (
            "convex",
            0.5,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            [[0.0, 0.0, 0.0]],
        ),
    ],
)
def test_time_decay_knn_empty(mat_no_timestamps, decay_fn, decay_coeff, expected_similarities, expected_out):

    algo = TimeDecayingNearestNeighbour(K=2, decay_coeff=decay_coeff, decay_fn=decay_fn, decay_interval=1)

    with pytest.warns(UserWarning) as record:
        algo.fit(mat_no_timestamps)

    assert str(record[0].message) == "TimeDecayingNearestNeighbour missing similar items for 3 items."

    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)

    # Make sure the similarities recommended are the similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))

    with pytest.warns(UserWarning):
        result = algo.predict(_in)
    assert str(record[0].message) == "TimeDecayingNearestNeighbour missing similar items for 3 items."

    np.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    with pytest.warns(UserWarning) as record:
        result = algo.predict(_in)
    assert str(record[0].message) == "TimeDecayingNearestNeighbour missing similar items for 3 items."

    np.testing.assert_almost_equal(result.toarray(), expected_out)


@pytest.mark.parametrize(
    "decay_fn, decay_coeff",
    [
        ("linear", 1.1),
        ("linear", -0.1),
        ("concave", 1.1),
        ("concave", -0.1),
        ("convex", 1),
        ("convex", 0),
    ],
)
def test_time_decay_knn_coeff_validation(mat, decay_fn, decay_coeff):
    with pytest.raises(ValueError):
        TimeDecayingNearestNeighbour(decay_coeff=decay_coeff, decay_fn=decay_fn)


@pytest.mark.parametrize(
    "decay_fn",
    [
        "linearity",
        "cosine",
        "crap",
    ],
)
def test_time_decay_knn_fn_validation_error(decay_fn):
    with pytest.raises(ValueError):
        TimeDecayingNearestNeighbour(decay_fn=decay_fn)


@pytest.mark.parametrize(
    "decay_fn, decay_coeff, expected_similarities, expected_out",
    [
        (
            "linear",
            0.5,
            np.array([[0.0, 2 / 3, 1 / 2], [2 / 3, 0.0, 5 / 6], [1 / 2, 5 / 6, 0.0]]),
            [[2 / 3, 2 / 3, 4 / 3]],
        ),
    ],
)
def test_time_decay_knn_predict(mat, decay_fn, decay_coeff, expected_similarities, expected_out):

    algo = TimeDecayingNearestNeighbour(K=2, decay_coeff=decay_coeff, decay_fn=decay_fn, decay_interval=1)

    algo.fit(mat)
    assert type(algo.similarity_matrix_) is scipy.sparse.csr_matrix
    algo.similarity_matrix_ = algo.similarity_matrix_.toarray()
    assert type(algo.similarity_matrix_) is np.ndarray

    # Make sure the similarities recommended are the similarities as computed.
    # If we create users with a single item seen in order.
    _in = scipy.sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)
    assert type(algo.similarity_matrix_) is np.ndarray
    assert type(result) is scipy.sparse.csr_matrix


@pytest.mark.parametrize("interval", [20, 3600, 24 * 3600])
def test_decay_interval(interval):
    algo_1 = TimeDecayingNearestNeighbour(K=2, decay_interval=1)
    algo_2 = TimeDecayingNearestNeighbour(K=2, decay_interval=interval)

    input_matrix_1 = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])
    input_matrix_2 = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]]) * interval

    concave_decayed_1 = algo_1._concave_matrix_decay(input_matrix_1, 5)
    concave_decayed_2 = algo_2._concave_matrix_decay(input_matrix_2, 5 * interval)

    np.testing.assert_array_equal(concave_decayed_1.toarray(), concave_decayed_2.toarray())

    convex_decayed_1 = algo_1._convex_matrix_decay(input_matrix_1, 5)
    convex_decayed_2 = algo_2._convex_matrix_decay(input_matrix_2, 5 * interval)

    np.testing.assert_array_equal(convex_decayed_1.toarray(), convex_decayed_2.toarray())

    linear_decayed_1 = algo_1._linear_matrix_decay(input_matrix_1, 5)
    linear_decayed_2 = algo_2._linear_matrix_decay(input_matrix_2, 5 * interval)

    np.testing.assert_array_equal(linear_decayed_1.toarray(), linear_decayed_2.toarray())


@pytest.mark.parametrize("decay_interval", [0, 0.5])
def test_decay_interval_validation(decay_interval):
    with pytest.raises(ValueError):
        TimeDecayingNearestNeighbour(decay_interval=decay_interval)
