import recpack.evaluate.metrics as metrics
import recpack.helpers
import pandas as pd
import numpy
import scipy.sparse

import pytest


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {'userId': [1, 1, 1, 0, 0, 0], 'movieId': [1, 3, 4, 0, 2, 4], 'values': [1, 2, 1, 1, 1, 2]}

    matrix = scipy.sparse.csr_matrix((input_dict['values'], (input_dict['userId'], input_dict['movieId'])))
    return matrix


@pytest.mark.parametrize(
    "propensity_type, expected_score",
    [
        (metrics.PropensityType.UNIFORM, (2/3)),
        (metrics.PropensityType.GLOBAL, 0.6),
        (metrics.PropensityType.USER, (2/3)),
    ]
)
def test_SNIPS(propensity_type, expected_score):
    K = 2
    # Test uniform propensities
    factory = metrics.SNIPS_factory(propensity_type)

    data = generate_data()
    factory.fit(data)
    # assert that the propensities add up to 1 (they are supposed to be percentages)
    numpy.testing.assert_almost_equal(factory.propensities.sum(axis=1), 1)

    metric = factory.create(K)

    pred_users, pred_items, pred_values = [0, 0, 0, 1, 1, 1], [0, 2, 3, 1, 3, 4], [0.3, 0.2, 0.1, 0.23, 0.3, 0.5]
    true_users, true_items = [0, 0, 1, 1, 1], [0, 2, 0, 1, 3]
    pred = scipy.sparse.csr_matrix((pred_values, (pred_users, pred_items)), shape=(2, 5))
    true_data = scipy.sparse.csr_matrix(([1 for i in range(len(true_users))], (true_users, true_items)), shape=(2, 5))

    metric.update(pred.toarray(), true_data, [0, 1])
    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_score)


def test_recall():
    K = 2
    metric = metrics.RecallK(K)

    pred_users, pred_items, pred_values = [0, 0, 0, 1, 1, 1], [0, 2, 3, 1, 3, 4], [0.3, 0.2, 0.1, 0.23, 0.3, 0.5]
    true_users, true_items = [0, 0, 1, 1, 1], [0, 2, 0, 1, 3]
    pred = scipy.sparse.csr_matrix((pred_values, (pred_users, pred_items)), shape=(2, 5))
    true_data = scipy.sparse.csr_matrix(([1 for i in range(len(true_users))], (true_users, true_items)), shape=(2, 5))

    metric.update(pred.toarray(), true_data)

    assert metric.num_users == 2
    assert metric.value == 0.75
