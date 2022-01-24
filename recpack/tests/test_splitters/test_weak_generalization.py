import numpy as np
import pytest

import recpack.splitters.scenarios as scenarios


def test_weak_generalization_invalid_perc(data_m):
    with pytest.raises(AssertionError):
        scenarios.WeakGeneralization(
            0.7, frac_interactions_validation=0.5, validation=True
        )


@pytest.mark.parametrize("frac_interactions_train", [0.5, 0.7])
def test_weak_generalization_split(data_m, frac_interactions_train):

    scenario = scenarios.WeakGeneralization(frac_interactions_train)
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    # Test no validation data
    with pytest.raises(KeyError):
        scenario.validation_data
    # Test approximately correct split
    perc_interactions_test = 1 - frac_interactions_train

    np.testing.assert_almost_equal(
        tr.values.nnz / data_m.values.nnz, frac_interactions_train, decimal=2
    )
    np.testing.assert_almost_equal(
        te_data_out.values.nnz / data_m.values.nnz, perc_interactions_test, decimal=2
    )

    # te_data_in =~ tr (except users that had no interactions in te_data_out)
    assert set(tr.indices[0]) == set(te_data_in.indices[0])

    # Users have interactions in both
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize(
    "frac_interactions_train, frac_interactions_validation", [(0.5, 0.25), (0.25, 0.25)]
)
def test_weak_generalization_split_w_validation(
    data_m, frac_interactions_train, frac_interactions_validation
):

    scenario = scenarios.WeakGeneralization(
        frac_interactions_train,
        frac_interactions_validation=frac_interactions_validation,
        validation=True,
    )
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data
    # Test approximately correct split
    perc_interactions_test = 1 - frac_interactions_train - frac_interactions_validation

    np.testing.assert_almost_equal(
        tr.values.nnz / data_m.values.nnz, frac_interactions_train, decimal=2
    )
    np.testing.assert_almost_equal(
        te_data_out.values.nnz / data_m.values.nnz, perc_interactions_test, decimal=2
    )
    np.testing.assert_almost_equal(
        val_data_out.values.nnz / data_m.values.nnz,
        frac_interactions_validation,
        decimal=2,
    )

    # Users have interactions in both
    assert te_data_out.active_users == te_data_in.active_users
    assert val_data_in.active_users == val_data_out.active_users

    # tr = val_data_in
    assert val_data_in.active_users == tr.active_users
    assert val_data_in.values.nnz == tr.values.nnz


def test_weak_generalization_mismatching_train_validation_in(data_m_sporadic_users):
    """Test with special case,
    where some users have too few events to be split into each of the slices.

    A user with a single event will only appear in the training dataset.
    Users with only 2 events will not occur in validation_in (and validation_out).
    Once a user has 3 events, they will appear in all datasets.

    We explicitly check train, validation_in and test_in data.
    Because train and validation would be expected to be the same,
    were it not for those that have to few interactions to appear in both.
    """
    # no parametrization of these values,
    # this way we know how much items are in each slice of the data.
    frac_interactions_train = 0.25
    frac_interactions_validation = 0.25

    scenario = scenarios.WeakGeneralization(
        frac_interactions_train,
        frac_interactions_validation=frac_interactions_validation,
        validation=True,
    )
    scenario.split(data_m_sporadic_users)

    tr = scenario.training_data
    te_data_in, _ = scenario.test_data
    val_data_in, _ = scenario.validation_data

    # Not all users in training are also in val_data_in
    assert val_data_in.active_users != tr.active_users

    temp = data_m_sporadic_users._df.groupby("uid").iid.count().reset_index()
    users_expected_in_all = temp[temp.iid >= 3].uid.unique()
    users_expected_in_test_and_train = temp[temp.iid >= 2].uid.unique()

    # Validation is the smallest subset
    assert set(val_data_in.active_users) == set(users_expected_in_all)
    # Test is a superset of validation
    assert set(te_data_in.active_users) == set(users_expected_in_test_and_train)
    # Train should contain all users
    assert tr.active_users == data_m_sporadic_users.active_users


def test_weak_generalization_mismatching_train_test_in(data_m_sporadic_users):
    """Test with special case,
    where some users have too few events to be split into each of the slices.

    A user with a single event will only appear in the training dataset.
    Users with 2 or more events in both train and test_in dataset
    """
    # no parametrization of these values,
    # this way we know how much items are in each slice of the data.
    frac_interactions_train = 0.50

    scenario = scenarios.WeakGeneralization(
        frac_interactions_train,
        validation=False,
    )
    scenario.split(data_m_sporadic_users)

    tr = scenario.training_data
    te_data_in, _ = scenario.test_data

    # Not all users in training are also in val_data_in
    assert te_data_in.active_users != tr.active_users

    temp = data_m_sporadic_users._df.groupby("uid").iid.count().reset_index()
    users_expected_in_test_and_train = temp[temp.iid >= 2].uid.unique()

    # Test is a subset
    assert set(te_data_in.active_users) == set(users_expected_in_test_and_train)
    # Train should contain all users
    assert tr.active_users == data_m_sporadic_users.active_users


@pytest.mark.parametrize(
    "frac_interactions_train, frac_interactions_validation", [(0.5, 0.25), (0.25, 0.25)]
)
def test_strong_generalization_timed_split_seed(
    data_m, frac_interactions_train, frac_interactions_validation
):

    # First scenario uses a random seed
    scenario_1 = scenarios.WeakGeneralization(
        frac_interactions_train,
        frac_interactions_validation=frac_interactions_validation,
        validation=True,
    )
    seed = scenario_1.seed
    scenario_1.split(data_m)

    # second scenario uses same seed as the previous one
    scenario_2 = scenarios.WeakGeneralization(
        frac_interactions_train,
        frac_interactions_validation=frac_interactions_validation,
        validation=True,
        seed=seed,
    )
    scenario_2.split(data_m)

    assert (
        scenario_1.training_data.num_interactions
        == scenario_2.training_data.num_interactions
    )

    assert (
        scenario_1.test_data_in.num_interactions
        == scenario_2.test_data_in.num_interactions
    )
    assert (
        scenario_1.test_data_out.num_interactions
        == scenario_2.test_data_out.num_interactions
    )
