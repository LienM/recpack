import recpack.metrics as metrics
import numpy
import scipy.sparse

import pytest


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {
        "userId": [1, 1, 1, 0, 0, 0],
        "movieId": [1, 3, 4, 0, 2, 4],
        "values": [1, 2, 1, 1, 1, 2],
    }

    matrix = scipy.sparse.csr_matrix(
        (input_dict["values"], (input_dict["userId"], input_dict["movieId"]))
    )
    return matrix


def test_UniformInversePropensity():
    data = generate_data()
    p = metrics.UniformInversePropensity(data)
    # Make sure the propensity computation is right
    numpy.testing.assert_almost_equal(p._get_propensities(data).sum(axis=1), 1)

    # Make sure the inverse propensities are right
    numpy.testing.assert_almost_equal(p.get([0, 1]), data.shape[1])


def test_GlobalInversePropensity():
    data = generate_data()

    p = metrics.GlobalInversePropensity(data)
    # Make sure the propensity computation is right
    numpy.testing.assert_almost_equal(p._get_propensities(data).sum(axis=1), 1)

    # Make sure the inverse propensities are right
    numpy.testing.assert_almost_equal(
        p.get([0, 1]), numpy.array([[8, 8, 8, 8 / 2, 8 / 3]])
    )


def test_UserInversePropensity():
    data = generate_data()
    users = [0, 1]
    p = metrics.UserInversePropensity(data)
    # Make sure the propensity computation is right
    numpy.testing.assert_almost_equal(p._get_propensities(users).sum(axis=1), 1)

    # Make sure the inverse propensities are right
    numpy.testing.assert_almost_equal(
        p.get(users).toarray(), numpy.array([[4, 0, 4, 0, 4 / 2], [0, 4, 0, 4 / 2, 4]])
    )


@pytest.mark.parametrize(
    "propensity_type, expected_class",
    [
        (metrics.PropensityType.UNIFORM, metrics.UniformInversePropensity),
        (metrics.PropensityType.GLOBAL, metrics.GlobalInversePropensity),
        (metrics.PropensityType.USER, metrics.UserInversePropensity),
    ],
)
def test_SNIPS_factory(propensity_type, expected_class):
    data = generate_data()
    factory = metrics.SNIPSFactory(propensity_type)
    factory.fit(data)

    K = 2
    metric = factory.create(K)
    assert type(metric.inverse_propensities) == expected_class
    assert metric.K == K

    K_values = [1, 2, 3]
    metric_dict = factory.create_multipe_SNIPS(K_values)
    for K in K_values:
        name = f"SNIPS@{K}"
        assert name in metric_dict
        assert type(metric_dict[name].inverse_propensities) == expected_class
        assert metric_dict[name].K == K


@pytest.mark.parametrize(
    "propensity_type, expected_score",
    [
        (metrics.PropensityType.UNIFORM, (2 / 3)),
        (metrics.PropensityType.GLOBAL, 0.6),
        (metrics.PropensityType.USER, (2 / 3)),
    ],
)
def test_SNIPS(propensity_type, expected_score):
    K = 2
    # Test uniform propensities
    factory = metrics.SNIPSFactory(propensity_type)

    data = generate_data()
    factory.fit(data)
    metric = factory.create(K)

    pred_users, pred_items, pred_values = (
        [0, 0, 0, 1, 1, 1],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )
    true_users, true_items = [0, 0, 1, 1, 1], [0, 2, 0, 1, 3]
    pred = scipy.sparse.csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(2, 5)
    )
    true_data = scipy.sparse.csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(2, 5)
    )

    metric.update(pred, true_data, [0, 1])
    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_score)


def test_recall():
    K = 2
    metric = metrics.RecallK(K)

    pred_users, pred_items, pred_values = (
        [0, 0, 0, 1, 1, 1],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )
    true_users, true_items = [0, 0, 1, 1, 1], [0, 2, 0, 1, 3]
    pred = scipy.sparse.csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(2, 5)
    )
    true_data = scipy.sparse.csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(2, 5)
    )

    metric.update(pred, true_data)

    assert metric.num_users == 2
    assert metric.value == 0.75
