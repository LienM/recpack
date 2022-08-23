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


@pytest.mark.parametrize(
    "decay_function",
    [
        ExponentialDecay,
        LinearDecay,
    ],
)
def test_disabled_with_0(decay_function):
    df = decay_function(0)
    output = df(np.array([0.5, 1, 2]))

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize(
    "decay_function",
    [ConvexDecay],
)
def test_disabled_with_1(decay_function):
    df = decay_function(1)
    output = df(np.array([0.5, 1, 2]))

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize("decay_function", [ExponentialDecay, ConvexDecay, ConcaveDecay, LinearDecay])
def test_order_preservation(decay_function):
    for i in range(100):
        a = np.random.randint(100)
        b = np.random.randint(100)

        decay = np.random.rand()
        df = decay_function(decay)

        output = df(np.array([a, b]))
        # Older events should get less weight
        assert (output[0] < output[1]) == (a > b)


def test_order_preservation_log_decay():
    for i in range(100):
        a = np.random.randint(100)
        b = np.random.randint(100)

        decay = np.random.randint(2, 10)

        output = LogDecay(decay)(np.array([a, b]))
        # Older events should get less weight
        assert (output[0] < output[1]) == (a > b)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([1, 1]), 0, np.array([1, 1])),
        (np.array([0.5, 1]), 0, np.array([1, 1])),
        (np.array([4, 2]), 1 / 2, np.array([np.exp(-2), np.exp(-1)])),
        (np.array([0, 3]), 1 / 2, np.array([1, np.exp(-3 / 2)])),
    ],
)
def test_exponential_decay(input, decay, expected_output):
    result = ExponentialDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([1, 1]), 0, np.array([1, 1])),
        (np.array([0.5, 1]), 0, np.array([1, 1])),
        (np.array([4, 2]), 1 / 2, np.array([0.5, 0.75])),
        (np.array([0, 3]), 1, np.array([1, 0])),
    ],
)
def test_linear_decay(input, decay, expected_output):
    result = LinearDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([4, 2]), 1 / 2, np.array([0.5 ** 4, 0.5 ** 2])),
        (np.array([0, 3]), 1 / 2, np.array([0.5 ** 0, 0.5 ** 3])),
    ],
)
def test_convex_decay(input, decay, expected_output):
    result = ConvexDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([4, 2]), 1 / 2, np.array([1 - 0.5 ** 0, 1 - 0.5 ** (1 - 0.5)])),
        (np.array([1, 3]), 0.3, np.array([1 - 0.3 ** (1 - 1 / 3), 1 - 0.3 ** 0])),
    ],
)
def test_concave_decay(input, decay, expected_output):
    result = ConcaveDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([1, 1]), 2, np.array([0, 0])),
        (np.array([0.4, 1]), 2, np.array([0.2, 0])),
    ],
)
def test_linear_decay_steeper(input, decay, expected_output):
    result = LinearDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([1, 1]), 1, np.array([1, 1])),
        (np.array([7200, 3600]), 3600, np.array([1 / 7200, 1 / 3600])),
        (np.array([0, 1]), 1, np.array([1, 1])),
    ],
)
def test_inverse_decay(input, decay, expected_output):
    result = InverseDecay(decay)(input)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "decay_function, input, decay, max_age, expected_output",
    [
        (ExponentialDecay, np.array([1, 2, 3]), 0.5, 6, np.exp(np.array([-0.5, -1, -1.5]))),
        (ConvexDecay, np.array([1, 2, 3]), 0.5, 6, np.array([0.5 ** 1, 0.5 ** 2, 0.5 ** 3])),
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
def test_max_age(decay_function, input, decay, max_age, expected_output):
    output = decay_function(decay)(input, max_age=max_age)
    np.testing.assert_array_almost_equal(output, expected_output)
