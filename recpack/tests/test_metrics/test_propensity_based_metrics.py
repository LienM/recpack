import numpy
import pytest
import scipy.sparse

import recpack.metrics as metrics


def test_UniformInversePropensity(data):
    p = metrics.UniformInversePropensity(data)
    # Make sure the propensity computation is right
    numpy.testing.assert_almost_equal(p._get_propensities(data).sum(axis=1), 1)

    # Make sure the inverse propensities are right
    numpy.testing.assert_almost_equal(p.get([0, 1]), data.shape[1])


def test_GlobalInversePropensity(data):
    p = metrics.GlobalInversePropensity(data)
    # Make sure the propensity computation is right
    numpy.testing.assert_almost_equal(p._get_propensities(data).sum(axis=1), 1)

    # Make sure the inverse propensities are right
    numpy.testing.assert_almost_equal(
        p.get([0, 1]), numpy.array([[8, 8, 8, 8 / 2, 8 / 3]])
    )


def test_UserInversePropensity(data):
    users = [0, 2]
    p = metrics.UserInversePropensity(data)
    # Make sure the propensity computation is right
    numpy.testing.assert_almost_equal(p._get_propensities(users).sum(axis=1)[users], 1)

    expected_values = [4, 0, 4, 0, 4 / 2, 0, 4, 0, 4 / 2, 4]
    expected_users = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
    expected_items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    expected_mat = scipy.sparse.csr_matrix((expected_values, (expected_users, expected_items)), shape=(10, 5))
    # Make sure the inverse propensities are right
    numpy.testing.assert_almost_equal(
        p.get(users).toarray(), expected_mat.toarray()
    )


@pytest.mark.parametrize(
    "propensity_type, expected_class",
    [
        (metrics.PropensityType.UNIFORM, metrics.UniformInversePropensity),
        (metrics.PropensityType.GLOBAL, metrics.GlobalInversePropensity),
        (metrics.PropensityType.USER, metrics.UserInversePropensity),
    ],
)
def test_SNIPS_factory(data, propensity_type, expected_class):
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
def test_SNIPS(data, X_pred, X_true, propensity_type, expected_score):
    K = 2


    # Test uniform propensities
    factory = metrics.SNIPSFactory(propensity_type)

    factory.fit(data)
    metric = factory.create(K)

    metric.update(X_pred, X_true)
    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_score)


@pytest.mark.parametrize(
    "propensity_type, expected_score, batch_size",
    [
        (metrics.PropensityType.UNIFORM, (2 / 3), 1),
        (metrics.PropensityType.GLOBAL, 0.6, 1),
        (metrics.PropensityType.USER, (2 / 3), 1),
    ],
)
def test_SNIPS_w_batch_size(data, X_pred, X_true, propensity_type, expected_score, batch_size):

    K = 2
    # Test uniform propensities
    factory = metrics.SNIPSFactory(propensity_type)

    factory.fit(data)
    metric = factory.create(K)

    max_user = X_pred.shape[0]
    u_c = 0

    while u_c < max_user:
        u_end = min(u_c + batch_size, max_user)
        users = list(range(u_c, u_end))

        mask = numpy.zeros((X_true.shape[0], 1))
        mask[users, 0] = 1

        local_pred = X_pred.multiply(mask).tocsr()
        local_pred.eliminate_zeros()
        local_true = X_true.multiply(mask).tocsr()

        if local_pred.nnz > 0:
            metric.update(local_pred, local_true)
        u_c = u_end

    assert metric.num_users == 2
    numpy.testing.assert_almost_equal(metric.value, expected_score)
