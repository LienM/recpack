import numpy as np
import pytest

from recpack.algorithms.time_aware_item_knn.decay_functions import (
    exponential_decay,
    log_decay,
    linear_decay,
    convex_decay,
    concave_decay,
)


@pytest.mark.parametrize(
    "decay_function",
    [
        exponential_decay,
        linear_decay,
    ],
)
def test_disabled_with_0(decay_function):
    output = decay_function(np.array([0.5, 1, 2]), 0)

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize(
    "decay_function",
    [convex_decay],
)
def test_disabled_with_1(decay_function):
    output = decay_function(np.array([0.5, 1, 2]), 1)

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize("decay_function", [exponential_decay, convex_decay, concave_decay, linear_decay])
def test_order_preservation(decay_function):
    for i in range(100):
        a = np.random.randint(100)
        b = np.random.randint(100)

        decay = np.random.rand()

        output = decay_function(np.array([a, b]), decay)
        print(a, b, decay)
        # Older events should get less weight
        assert (output[0] < output[1]) == (a > b)


def test_order_preservation_log_decay():
    for i in range(100):
        a = np.random.randint(100)
        b = np.random.randint(100)

        decay = np.random.randint(2, 10)

        output = log_decay(np.array([a, b]), decay)
        print(a, b, decay)
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
    result = exponential_decay(input, decay)
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
    result = linear_decay(input, decay)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([4, 2]), 1 / 2, np.array([0.5 ** 4, 0.5 ** 2])),
        (np.array([0, 3]), 1 / 2, np.array([0.5 ** 0, 0.5 ** 3])),
    ],
)
def test_convex_decay(input, decay, expected_output):
    result = convex_decay(input, decay)
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "input, decay, expected_output",
    [
        (np.array([4, 2]), 1 / 2, np.array([1 - 0.5 ** 0, 1 - 0.5 ** (1 - 0.5)])),
        (np.array([1, 3]), 0.3, np.array([1 - 0.3 ** (1 - 1 / 3), 1 - 0.3 ** 0])),
    ],
)
def test_concave_decay(input, decay, expected_output):
    result = concave_decay(input, decay)
    np.testing.assert_array_almost_equal(result, expected_output)
