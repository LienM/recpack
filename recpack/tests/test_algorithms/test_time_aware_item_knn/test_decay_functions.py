import numpy as np
import pytest

from recpack.algorithms.time_aware_item_knn.decay_functions import (
    exponential_decay,
    log_decay,
    concave_decay,
    convex_decay,
    linear_decay,
)


@pytest.mark.parametrize("func", [exponential_decay, linear_decay])
def test_disabled_with_0(func):
    output = func(np.array([0.5, 1, 2]), 0)

    # With decay = 0, the output should be a binary vector.
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize("func", [exponential_decay, convex_decay, concave_decay, linear_decay, log_decay])
def test_order_preservation(func):
    for i in range(100):
        a = np.random.randint(100)
        b = np.random.randint(100)

        # TODO: make sure decay is not 0
        decay = np.random.rand()

        output = func(np.array([a, b]), decay)
        print(a, b, decay)
        # Older events should get less weight
        assert (output[0] < output[1]) == (a > b)


def test_order_preservation_log():
    for i in range(100):
        a = np.random.rand()
        b = np.random.rand()

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
        (np.array([1, 1]), 0, np.array([0, 0])),
        (np.array([0.5, 1]), 0, np.array([0, 0])),
        (np.array([4, 2]), 1 / 2, np.array([0.5 ** 4, 0.5 ** 2])),
        (np.array([0, 3]), 1 / 2, np.array([0.5 ** 0, 0.5 ** 3])),
    ],
)
def test_convex_decay(input, decay, expected_output):
    result = convex_decay(input, decay)
    np.testing.assert_array_almost_equal(result, expected_output)
