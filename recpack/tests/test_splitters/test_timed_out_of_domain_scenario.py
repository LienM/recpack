import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split(data_m, t):

    scenario = scenarios.TimedOutOfDomainPredictAndEvaluate(t)
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()

    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split_w_validation_no_validation_t(data_m, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.TimedOutOfDomainPredictAndEvaluate(t, validation=True)


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split_w_validation_validation_t_too_large(data_m, t):
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.TimedOutOfDomainPredictAndEvaluate(t, t_validation=t, validation=True)


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split_validation(data_m, t):
    t_validation = t - 10
    scenario = scenarios.TimedOutOfDomainPredictAndEvaluate(
        t, t_validation=t_validation, validation=True
    )
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert (tr.timestamps < t_validation).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert (val_data_in.timestamps < t_validation).all()
    assert (val_data_out.timestamps < t).all()
    assert (val_data_out.timestamps >= t_validation).all()

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split_w_validation_has_validation_users(data_m, t):
    t_validation = t - 10
    scenario = scenarios.TimedOutOfDomainPredictAndEvaluate(
        t, t_validation=t_validation, validation=True
    )
    scenario.split(data_m, data_m)

    val_data_in, val_data_out = scenario.validation_data

    assert val_data_out.active_user_count > 0
    assert val_data_in.active_user_count > 0


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_evaluate(data_m, t):

    scenario = scenarios.TrainInTimedOutOfDomainEvaluate(t)
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()

    # Training is completely input for test
    assert (tr.values.nonzero()[0] == te_data_in.values.nonzero()[0]).all()
    assert (tr.values.nonzero()[1] == te_data_in.values.nonzero()[1]).all()

    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_evaluate_validate(data_m, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.TrainInTimedOutOfDomainEvaluate(t, validation=True)
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.TrainInTimedOutOfDomainEvaluate(t, t_validation=t, validation=True)

    t_validation = t - 10
    scenario = scenarios.TrainInTimedOutOfDomainEvaluate(
        t, t_validation=t_validation, validation=True
    )
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert (tr.timestamps < t_validation).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert (val_data_in.timestamps < t_validation).all()
    assert (val_data_out.timestamps < t).all()
    assert (val_data_out.timestamps >= t_validation).all()

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_evaluate_labels(data_m, t):

    scenario = scenarios.TrainInTimedOutOfDomainWithLabelsEvaluate(t)
    scenario.split(data_m, data_m)

    tr, train_y = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (train_y.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()

    # Training is completely input for test
    assert (tr.values.nonzero()[0] == te_data_in.values.nonzero()[0]).all()
    assert (tr.values.nonzero()[1] == te_data_in.values.nonzero()[1]).all()

    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_evaluate_labels_validate(data_m, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.TrainInTimedOutOfDomainWithLabelsEvaluate(t, validation=True)
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.TrainInTimedOutOfDomainWithLabelsEvaluate(
            t, t_validation=t, validation=True
        )

    t_validation = t - 10
    scenario = scenarios.TrainInTimedOutOfDomainWithLabelsEvaluate(
        t, t_validation=t_validation, validation=True
    )
    scenario.split(data_m, data_m)

    tr, train_y = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data

    assert (tr.timestamps < t_validation).all()
    assert (train_y.timestamps < t_validation).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()
    assert (val_data_in.timestamps < t_validation).all()
    assert (val_data_out.timestamps < t).all()
    assert (val_data_out.timestamps >= t_validation).all()

    # This should be guaranteed given that data_1 and data_2 are identical

    assert set(val_data_in.indices[0]) == set(val_data_out.indices[0])

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users
