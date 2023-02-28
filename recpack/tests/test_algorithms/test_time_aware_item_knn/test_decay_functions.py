# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pytest

from recpack.algorithms.time_aware_item_knn.decay_functions import (
    ExponentialDecay,
    LogDecay,
    LinearDecay,
    ConvexDecay,
    ConcaveDecay,
    InverseDecay,
)


DUMMY_ARRAY = np.array([0.5, 1, 2])


@pytest.mark.parametrize(
    "decay_function",
    [
        ExponentialDecay(0),
        LinearDecay(0, DUMMY_ARRAY.max()),
    ],
)
def test_disabled_with_0(decay_function):
    output = decay_function(DUMMY_ARRAY)

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize(
    "decay_function",
    [ConvexDecay(1)],
)
def test_disabled_with_1(decay_function):
    output = decay_function(DUMMY_ARRAY)

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


MAX_DISTANCE = 100


@pytest.mark.parametrize(
    "decay_function",
    [ExponentialDecay(np.random.rand()) for _ in range(10)]
    + [ConvexDecay(np.random.rand()) for _ in range(10)]
    + [ConcaveDecay(np.random.rand(), MAX_DISTANCE) for _ in range(10)]
    + [LinearDecay(np.random.rand(), MAX_DISTANCE) for _ in range(10)]
    + [LogDecay(np.random.randint(2, 10), MAX_DISTANCE) for _ in range(10)],
)
def test_order_preservation(decay_function):
    for i in range(100):
        a = np.random.randint(MAX_DISTANCE)
        b = np.random.randint(MAX_DISTANCE)

        output = decay_function(np.array([a, b]))
        # Older events should get less weight
        assert (output[0] < output[1]) == (a > b)


# fmt: off
@pytest.mark.parametrize(
    "input, decay_function, expected_output",
    [
        (np.array([1, 1]), ExponentialDecay(0), np.array([1, 1])),
        (np.array([0.5, 1]), ExponentialDecay(0), np.array([1, 1])),
        (np.array([4, 2]), ExponentialDecay(1 / 2), np.array([np.exp(-2), np.exp(-1)])),
        (np.array([0, 3]), ExponentialDecay(1 / 2), np.array([1, np.exp(-3 / 2)])),

        (np.array([1, 1]), LinearDecay(0, 1), np.array([1, 1])),
        (np.array([0.5, 1]), LinearDecay(0, 1), np.array([1, 1])),
        (np.array([4, 2]), LinearDecay(1 / 2, 4), np.array([0.5, 0.75])),
        (np.array([0, 3]), LinearDecay(1, 3), np.array([1, 0])),
        (np.array([1, 1]), LinearDecay(2, 1), np.array([0, 0])),
        (np.array([0.4, 1]), LinearDecay(2, 1), np.array([0.2, 0])),
                                                     
        (np.array([4, 2]), ConvexDecay(1 / 2), np.array([0.5 ** 4, 0.5 ** 2])),
        (np.array([0, 3]), ConvexDecay(1 / 2), np.array([0.5 ** 0, 0.5 ** 3])),

        (np.array([4, 2]), ConcaveDecay(0.5, 4), np.array([1 - 0.5 ** 0, 1 - 0.5 ** (1 - 0.5)])),
        (np.array([1, 3]), ConcaveDecay(0.3, 3), np.array([1 - 0.3 ** (1 - 1 / 3), 1 - 0.3 ** 0])),

        (np.array([1, 1]), InverseDecay(), np.array([1, 1])),
        (np.array([7200, 3600]), InverseDecay(), np.array([1 / 7200, 1 / 3600])),
        (np.array([0, 1]), InverseDecay(), np.array([1, 1])),

    ],
)
def test_decay_computation(input, decay_function, expected_output):
    result = decay_function(input)
    np.testing.assert_array_almost_equal(result, expected_output)

# fmt: on


@pytest.mark.parametrize(
    "decay_function, input, decay, max_distance, expected_output",
    [
        (
            ConcaveDecay,
            np.array([1, 2, 3]),
            0.5,
            6,
            np.array([1 - 0.5 ** (1 - 1 / 6), 1 - 0.5 ** (1 - 2 / 6), 1 - 0.5 ** (1 - 3 / 6)]),
        ),
        (
            LogDecay,
            np.array([1, 2, 3]),
            np.e,
            6,
            np.array(
                [
                    np.log((np.e - 1) * (1 - 1 / 6) + 1),
                    np.log((np.e - 1) * (1 - 2 / 6) + 1),
                    np.log((np.e - 1) * (1 - 3 / 6) + 1),
                ]
            ),
        ),
        (
            LinearDecay,
            np.array([1, 2, 3]),
            0.5,
            6,
            np.array([1 - (1 / 6) * 0.5, 1 - (2 / 6) * 0.5, 1 - (3 / 6) * 0.5]),
        ),
    ],
)
def test_max_distance(decay_function, input, decay, max_distance, expected_output):
    output = decay_function(decay, max_distance)(input)
    np.testing.assert_array_almost_equal(output, expected_output)


@pytest.mark.parametrize("decay_function", [LinearDecay(1, 1), LogDecay(2, 1), ConcaveDecay(0.5, 1)])
def test_wrong_max_distance(decay_function):
    with pytest.raises(ValueError) as e:
        decay_function(DUMMY_ARRAY)

    assert e.match("At least one of the distances is bigger than the specified max_distance.")


@pytest.mark.parametrize(
    "decay_function, decay_value",
    [
        (ExponentialDecay, 0),
        (ExponentialDecay, 1),
        (ExponentialDecay, 0.5),
        (ConvexDecay, 1),
        (ConvexDecay, 0.5),
        (ConcaveDecay, 1),
        (ConcaveDecay, 0.5),
        (LogDecay, 2),
        (LogDecay, 4),
        (LogDecay, 100),
        (LinearDecay, 0),
        (LinearDecay, 2),
    ],
)
def test_validation_succeeds(decay_function, decay_value):
    # If the validation fails, the function will raise an error
    decay_function.validate_decay(decay_value)


@pytest.mark.parametrize(
    "decay_function, decay_value",
    [
        (ExponentialDecay, 2),
        (ExponentialDecay, -1),
        (ExponentialDecay, 1.000001),
        (ConvexDecay, 0),
        (ConvexDecay, 2),
        (ConvexDecay, -1),
        (ConcaveDecay, 0),
        (ConcaveDecay, 2),
        (ConcaveDecay, -1),
        (LogDecay, 0),
        (LogDecay, 1),
        (LogDecay, -1),
        (LinearDecay, -1),
    ],
)
def test_validation_fails(decay_function, decay_value):
    with pytest.raises(ValueError) as e:
        decay_function.validate_decay(decay_value)

    assert e.match("Decay parameter = ")
