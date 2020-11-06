import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split(data_m, perc_users_in, t):

    scenario = scenarios.StrongGeneralizationTimed(perc_users_in, t)
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation_no_validation_t(data_m, perc_users_in, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.StrongGeneralizationTimed(perc_users_in, t, validation=True)


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation_validation_t_too_large(data_m, perc_users_in, t):
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.StrongGeneralizationTimed(
            perc_users_in, t, t_validation=t, validation=True
        )


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation(data_m, perc_users_in, t):
    t_validation = t - 10
    scenario = scenarios.StrongGeneralizationTimed(
        perc_users_in, t, t_validation=t_validation, validation=True
    )
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    assert (tr.timestamps < t_validation).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert (val_data_in.timestamps < t_validation).all()
    assert (val_data_out.timestamps < t).all()
    assert (val_data_out.timestamps >= t_validation).all()

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split_w_validation_has_validation_users(data_m, perc_users_in, t):
    t_validation = t - 10
    scenario = scenarios.StrongGeneralizationTimed(
        perc_users_in, t, t_validation=t_validation, validation=True
    )
    scenario.split(data_m)

    val_data_in, val_data_out = scenario.validation_data

    assert val_data_out.active_user_count > 0
    assert val_data_in.active_user_count > 0
