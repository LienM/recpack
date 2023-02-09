# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pandas as pd
import pytest

from recpack.matrix import InteractionMatrix
import recpack.scenarios as scenarios


@pytest.mark.parametrize("frac_data_in", [0.5, 0.7])
def test_weak_generalization_split(data_m, frac_data_in):

    scenario = scenarios.WeakGeneralization(frac_data_in)
    scenario.split(data_m)

    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    # Test no validation data
    with pytest.raises(KeyError):
        scenario.validation_data
    # Test approximately correct split
    frac_interactions_test = 1 - frac_data_in

    np.testing.assert_almost_equal(tr.num_interactions / data_m.num_interactions, frac_data_in, decimal=2)
    np.testing.assert_almost_equal(
        te_data_out.num_interactions / data_m.num_interactions,
        frac_interactions_test,
        decimal=2,
    )

    # te_data_in =~ tr (except users that had no interactions in te_data_out)
    assert set(tr.indices[0]) == set(te_data_in.indices[0])

    # Users have interactions in both
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("frac_data_in", [0.5, 0.25])
def test_weak_generalization_split_w_validation(larger_data_m, frac_data_in):
    # Test uses the larger_data_m fixture to improve stability of splitting.

    scenario = scenarios.WeakGeneralization(
        frac_data_in,
        validation=True,
    )
    scenario.split(larger_data_m)

    val_tr = scenario.validation_training_data
    full_tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    # Test approximately correct split
    frac_interactions_test = 1 - frac_data_in

    np.testing.assert_almost_equal(
        full_tr.num_interactions / larger_data_m.num_interactions,
        frac_data_in,
        decimal=2,
    )

    np.testing.assert_almost_equal(
        val_tr.num_interactions / full_tr.num_interactions,
        frac_data_in,
        decimal=2,
    )
    np.testing.assert_almost_equal(
        te_data_out.num_interactions / larger_data_m.num_interactions,
        frac_interactions_test,
        decimal=2,
    )
    np.testing.assert_almost_equal(
        val_data_out.num_interactions / full_tr.num_interactions,
        frac_interactions_test,
        decimal=2,
    )

    # te_data_in ~= tr (except users that had no interactions in te_data_out)
    assert set(val_tr.active_users).intersection(val_data_in.active_users) == val_data_in.active_users
    assert set(full_tr.active_users.intersection(te_data_in.active_users)) == te_data_in.active_users
    # Users have interactions in both
    assert te_data_out.active_users == te_data_in.active_users
    assert val_data_in.active_users == val_data_out.active_users

    # tr = val_data_in
    assert val_data_in.num_interactions == val_tr.num_interactions


def test_weak_generalization_mismatching_train_validation_in(
    data_m_sporadic_users,
):
    """Test with special case,
    where some users have too few events to be split into each of the slices.

    A user with a single event will only appear in the validation training dataset.
    Users with only 2 events will not occur in validation_in (and validation_out).
    Once a user has 3 events, they will appear in all datasets.

    We explicitly check train, validation_in and test_in data.
    Because train and validation would be expected to be the same,
    were it not for those that have to few interactions to appear in both.
    """
    # no parametrization of these values,
    # this way we know how much items are in each slice of the data.
    data_in_frac = 0.5

    scenario = scenarios.WeakGeneralization(
        data_in_frac,
        validation=True,
    )
    scenario.split(data_m_sporadic_users)

    tr = scenario.validation_training_data
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
    data_in_frac = 0.50

    scenario = scenarios.WeakGeneralization(
        data_in_frac,
        validation=False,
    )
    scenario.split(data_m_sporadic_users)

    tr = scenario.full_training_data
    te_data_in, _ = scenario.test_data

    # Not all users in training are also in val_data_in
    assert te_data_in.active_users != tr.active_users

    temp = data_m_sporadic_users._df.groupby("uid").iid.count().reset_index()
    users_expected_in_test_and_train = temp[temp.iid >= 2].uid.unique()

    # Test is a subset
    assert set(te_data_in.active_users) == set(users_expected_in_test_and_train)
    # Train should contain all users
    assert tr.active_users == data_m_sporadic_users.active_users


@pytest.mark.parametrize("frac_data_in", [0.5, 0.25])
def test_weak_generalization_timed_split_seed(data_m, frac_data_in):

    # First scenario uses a random seed
    scenario_1 = scenarios.WeakGeneralization(
        frac_data_in,
        validation=True,
    )
    seed = scenario_1.seed
    scenario_1.split(data_m)

    # second scenario uses same seed as the previous one
    scenario_2 = scenarios.WeakGeneralization(
        frac_data_in,
        validation=True,
        seed=seed,
    )
    scenario_2.split(data_m)

    assert scenario_1.full_training_data.num_interactions == scenario_2.full_training_data.num_interactions

    assert scenario_1.validation_training_data.num_interactions == scenario_2.validation_training_data.num_interactions

    assert scenario_1.test_data_in.num_interactions == scenario_2.test_data_in.num_interactions
    assert scenario_1.test_data_out.num_interactions == scenario_2.test_data_out.num_interactions

    assert scenario_1.validation_data_in.num_interactions == scenario_2.validation_data_in.num_interactions
    assert scenario_1.validation_data_out.num_interactions == scenario_2.validation_data_out.num_interactions


@pytest.fixture()
def interaction_matrix_different_history_sizes():
    input_dict = {
        InteractionMatrix.USER_IX: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        InteractionMatrix.ITEM_IX: [0, 1, 2, 0, 1, 2, 0, 1, 2, 3],
        InteractionMatrix.TIMESTAMP_IX: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    df = pd.DataFrame.from_dict(input_dict)
    data = InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    return data


def test_base_splitter(interaction_matrix_different_history_sizes):
    scen = scenarios.WeakGeneralization(0.66666, validation=True)

    scen.split(interaction_matrix_different_history_sizes)

    # User 0 has a single interaction which
    # should have ended up in the full training data set.
    assert scen.full_training_data.values[0].sum() == 1
    assert scen.test_data_out.values[0].sum() == 0

    # The single event should be filtered due to no event in the out dataset.
    assert scen.test_data_in.values[0].sum() == 0

    # The single event should have ended up in the validation_training dataset as well
    assert scen.validation_training_data.values[0].sum() == 1
    assert scen.validation_data_out.values[0].sum() == 0
    # And again filtered from the in dataset
    assert scen.validation_data_in.values[0].sum() == 0

    # User 1 has 2 interactions,
    # which should end up in full_training and validation training,
    # not enough events for an event in any of the out datasets
    assert scen.full_training_data.values[1].sum() == 2
    assert scen.test_data_out.values[1].sum() == 0
    assert scen.test_data_in.values[1].sum() == 0
    assert scen.validation_training_data.values[1].sum() == 2
    assert scen.validation_data_out.values[1].sum() == 0
    assert scen.validation_data_in.values[1].sum() == 0

    # User 2 has 3 interactions,
    # of which 2 should end up in full_training
    # 1 in test_out
    # 2 in validation training,
    # and none in the validation out dataset.
    assert scen.full_training_data.values[2].sum() == 2
    assert scen.test_data_out.values[2].sum() == 1
    assert scen.test_data_in.values[2].sum() == 2
    assert scen.validation_training_data.values[2].sum() == 2
    assert scen.validation_data_out.values[2].sum() == 0
    assert scen.validation_data_in.values[2].sum() == 0

    # User 3 has 4 interactions,
    # of which 3 should end up in full_training
    # 1 in test_out
    # 2 in validation training,
    # and 1 in the validation out dataset.
    assert scen.full_training_data.values[3].sum() == 3
    assert scen.test_data_out.values[3].sum() == 1
    assert scen.test_data_in.values[3].sum() == 3
    assert scen.validation_training_data.values[3].sum() == 2
    assert scen.validation_data_out.values[3].sum() == 1
    assert scen.validation_data_in.values[3].sum() == 2


@pytest.mark.parametrize("frac_data_in", [0.25, 0.5, 0.666, 0.8, 0.9])
def test_base_splitter_distributions(data_m, frac_data_in):
    scen = scenarios.WeakGeneralization(frac_data_in, validation=True)

    scen.split(data_m)

    np.testing.assert_almost_equal(
        scen.full_training_data.num_interactions / data_m.num_interactions,
        frac_data_in,
        2,
    )


# TODO: this test fails because user based sampling in WeakGeneralization
# does not guarantee the overall split, especially with sparse users.
# @pytest.mark.parametrize("frac_data_in", [0.25, 0.5, 0.666, 0.8, 0.9])
# def test_base_splitter_distributions_sparse_data(data_m_sparse, frac_data_in):
#     scen = scenarios.WeakGeneralization(frac_data_in, validation=True)

#     scen.split(data_m_sparse)

#     np.testing.assert_almost_equal(
#         scen.full_training_data.num_interactions / data_m_sparse.num_interactions,
#         frac_data_in,
#         2,
#     )
