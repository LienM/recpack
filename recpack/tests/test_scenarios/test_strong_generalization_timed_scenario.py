# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

import recpack.scenarios as scenarios


@pytest.mark.parametrize("frac_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split(data_m, frac_users_in, t):

    scenario = scenarios.StrongGeneralizationTimed(frac_users_in, t)
    scenario.split(data_m)

    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("frac_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation_no_validation_t(data_m, frac_users_in, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.StrongGeneralizationTimed(frac_users_in, t, validation=True)


@pytest.mark.parametrize("frac_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation_validation_t_too_large(data_m, frac_users_in, t):
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.StrongGeneralizationTimed(frac_users_in, t, t_validation=t, validation=True)


@pytest.mark.parametrize("frac_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation(data_m, frac_users_in, t):
    t_validation = t - 10
    scenario = scenarios.StrongGeneralizationTimed(frac_users_in, t, t_validation=t_validation, validation=True)
    scenario.split(data_m)

    val_tr = scenario.validation_training_data
    full_tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert not set(val_tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(val_tr.indices[0]).intersection(te_data_out.indices[0])
    assert not full_tr.active_users.intersection(te_data_out.active_users)
    assert full_tr.active_users.intersection(val_tr.active_users) == val_tr.active_users
    # Make sure all the users in validation_data_in are also in the training data.
    assert full_tr.active_users.intersection(val_data_in.active_users) == val_data_in.active_users

    assert (val_tr.timestamps < t_validation).all()
    assert (full_tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert (val_data_in.timestamps < t_validation).all()
    assert (val_data_out.timestamps < t).all()
    assert (val_data_out.timestamps >= t_validation).all()

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("frac_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation_has_validation_users(data_m, frac_users_in, t):
    t_validation = t - 10
    scenario = scenarios.StrongGeneralizationTimed(frac_users_in, t, t_validation=t_validation, validation=True)
    scenario.split(data_m)

    val_data_in, val_data_out = scenario.validation_data

    assert val_data_out.num_active_users > 0
    assert val_data_in.num_active_users > 0


@pytest.mark.parametrize("frac_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_seed(data_m, frac_users_in, t):

    t_validation = t - 10
    # First scenario uses a random seed
    scenario_1 = scenarios.StrongGeneralizationTimed(frac_users_in, t, t_validation=t_validation, validation=True)
    seed = scenario_1.seed
    scenario_1.split(data_m)

    # second scenario uses same seed as the previous one
    scenario_2 = scenarios.StrongGeneralizationTimed(
        frac_users_in, t, t_validation=t_validation, validation=True, seed=seed
    )
    scenario_2.split(data_m)

    assert scenario_1.full_training_data.num_interactions == scenario_2.full_training_data.num_interactions

    assert scenario_1.validation_training_data.num_interactions == scenario_2.validation_training_data.num_interactions

    assert scenario_1.test_data_in.num_interactions == scenario_2.test_data_in.num_interactions

    assert scenario_1.test_data_out.num_interactions == scenario_2.test_data_out.num_interactions
