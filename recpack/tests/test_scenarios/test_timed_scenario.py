# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

import recpack.scenarios as scenarios


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split(data_m, t):

    scenario = scenarios.Timed(t)
    scenario.split(data_m)

    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split_w_validation_no_validation_t(data_m, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.Timed(t, validation=True)


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split_w_validation_validation_t_too_large(data_m, t):
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.Timed(t, t_validation=t, validation=True)


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split_w_validation(data_m, t):
    t_validation = t - 10
    scenario = scenarios.Timed(t, t_validation=t_validation, validation=True)
    scenario.split(data_m)

    val_tr = scenario.validation_training_data
    full_tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert (val_tr.timestamps < t_validation).all()
    assert (full_tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert (val_data_in.timestamps < t_validation).all()
    assert (val_data_out.timestamps >= t_validation).all()
    assert (val_data_out.timestamps < t).all()

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split_w_validation_has_validation_users(data_m, t):
    t_validation = t - 10
    scenario = scenarios.Timed(t, t_validation=t_validation, validation=True)
    scenario.split(data_m)

    val_data_in, val_data_out = scenario.validation_data

    assert val_data_out.num_active_users > 0
    assert val_data_in.num_active_users > 0


def test_timed_split_w_validation_no_full_overlap_users(data_m_small):
    t = 9
    t_validation = 8
    scenario = scenarios.Timed(t, t_validation=t_validation, validation=True)
    scenario.split(data_m_small)

    # This should not change after the validation data fetch
    # This has been a bug which is fixed
    # using copy because the training data was a ref to the validation_data_in member
    # which is edited in the validation_data member fetching
    # Fix made it a copy internally, this test ensures this does not ever change back
    t_1 = scenario.full_training_data.copy()

    val_data_in, val_data_out = scenario.validation_data

    training_data = scenario.validation_training_data

    assert t_1.active_users == training_data.active_users
    assert training_data.active_users != val_data_in.active_users
