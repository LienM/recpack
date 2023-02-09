# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

import recpack.scenarios as scenarios


@pytest.mark.parametrize("frac_users_train, frac_interactions_in", [(0.7, 0.5), (0, 0.5)])
def test_strong_generalization_split(data_m, frac_users_train, frac_interactions_in):

    scenario = scenarios.StrongGeneralization(frac_users_train, frac_interactions_in)
    scenario.split(data_m)

    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    tr_users = set(tr.indices[0])
    te_in_users = set(te_data_in.indices[0])
    te_out_users = set(te_data_out.indices[0])
    te_users = te_in_users.union(te_out_users)

    te_in_interactions = te_data_in.indices[1]
    te_out_interactions = te_data_out.indices[1]

    # We expect the result to be approximately split, since it is random, it
    # is possible to not always be perfect.
    diff_allowed = 0.1

    assert abs(len(tr_users) / (len(tr_users) + len(te_users)) - frac_users_train) < diff_allowed

    # Higher volatility, so not as bad to miss
    diff_allowed = 0.2
    assert (
        abs(len(te_in_interactions) / (len(te_in_interactions) + len(te_out_interactions)) - frac_interactions_in)
        < diff_allowed
    )

    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("frac_users_train, frac_interactions_in", [(0.7, 0.5), (0.3, 0.5)])
def test_strong_generalization_split_w_validation(data_m, frac_users_train, frac_interactions_in):

    # Filter a bit in the data_m, so we only have users with at least 2
    # interactions.
    events_per_user = data_m.binary_values.sum(axis=1)
    # Users with only a single interaction -> NAY
    events_per_user[events_per_user == 1] = 0
    users = set(events_per_user.nonzero()[0])
    data_m.users_in(list(users), inplace=True)

    scenario = scenarios.StrongGeneralization(frac_users_train, frac_interactions_in, validation=True)
    scenario.split(data_m)

    val_tr = scenario.validation_training_data
    full_tr = scenario.full_training_data
    val_data_in, val_data_out = scenario.validation_data
    te_data_in, te_data_out = scenario.test_data

    val_tr_users = set(val_tr.indices[0])
    full_tr_users = set(full_tr.indices[0])
    te_in_users = set(te_data_in.indices[0])
    te_out_users = set(te_data_out.indices[0])
    val_in_users = set(val_data_in.indices[0])
    val_out_users = set(val_data_out.indices[0])

    assert not val_tr_users.intersection(te_in_users)
    assert not val_tr_users.intersection(val_in_users)
    assert not te_in_users.intersection(val_in_users)
    assert not full_tr_users.intersection(te_in_users)
    assert full_tr_users.intersection(val_in_users) == val_in_users
    assert full_tr_users.intersection(val_tr_users) == val_tr_users

    assert te_in_users == te_out_users
    assert val_in_users == val_out_users

    # We expect the result to be approximately split, since it is random, it
    # is possible to not always be perfect.
    diff_allowed = 0.1
    tr_to_val_perc = len(val_tr_users) / (len(val_tr_users) + len(val_in_users))
    # this is a non configurable value
    assert abs(tr_to_val_perc - 0.8) < diff_allowed

    tr_and_val_to_te_perc = (len(val_tr_users) + len(val_in_users)) / (
        len(val_tr_users) + len(val_in_users) + len(te_in_users)
    )
    assert abs(tr_and_val_to_te_perc - frac_users_train) < diff_allowed

    assert val_data_in.num_active_users > 0

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("frac_users_train, frac_interactions_in", [(0.7, 0.5), (0, 0.5)])
def test_strong_generalization_split_seed(data_m, frac_users_train, frac_interactions_in):

    # First scenario uses a random seed
    scenario_1 = scenarios.StrongGeneralization(frac_users_train, frac_interactions_in)
    seed = scenario_1.seed
    scenario_1.split(data_m)

    # second scenario uses same seed as the previous one
    scenario_2 = scenarios.StrongGeneralization(frac_users_train, frac_interactions_in, seed=seed)
    scenario_2.split(data_m)

    assert scenario_1.full_training_data.num_interactions == scenario_2.full_training_data.num_interactions

    assert scenario_1.test_data_in.num_interactions == scenario_2.test_data_in.num_interactions
    assert scenario_1.test_data_out.num_interactions == scenario_2.test_data_out.num_interactions
