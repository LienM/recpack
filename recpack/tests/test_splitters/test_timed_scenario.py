import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split(data_m, t):

    scenario = scenarios.Timed(t)
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_split_w_values(data_m_w_values, t):
    # Test covering bug removed from dataM

    scenario = scenarios.Timed(t)
    scenario.split(data_m_w_values)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert te_data_out.active_users == te_data_in.active_users

    assert tr.values.max() > 1


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

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert (tr.timestamps < t_validation).all()
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

    assert val_data_out.active_user_count > 0
    assert val_data_in.active_user_count > 0
