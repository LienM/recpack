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
def test_timed_split_validation(data_m, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.Timed(t, validation=True)
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.Timed(
            t, t_validation=t, validation=True)

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
def test_strong_generalization_timed_split_validation(
        data_m, perc_users_in, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.StrongGeneralizationTimed(perc_users_in, t, validation=True)
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.StrongGeneralizationTimed(perc_users_in,
                                            t, t_validation=t, validation=True)

    t_validation = t - 10
    scenario = scenarios.StrongGeneralizationTimed(
        perc_users_in, t, t_validation=t_validation,
        validation=True)
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
def test_timed_out_of_domain_split_validation(data_m, t):
    # Make sure exception is thrown if the validation timestamp is not provided
    with pytest.raises(Exception):
        scenarios.TimedOutOfDomainPredictAndEvaluate(t, validation=True)
    # Make sure t_validation < t
    with pytest.raises(AssertionError):
        scenarios.TimedOutOfDomainPredictAndEvaluate(
            t, t_validation=t, validation=True)

    t_validation = t - 10
    scenario = scenarios.TimedOutOfDomainPredictAndEvaluate(
        t, t_validation=t_validation, validation=True)
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
        scenarios.TrainInTimedOutOfDomainEvaluate(
            t, t_validation=t, validation=True)

    t_validation = t - 10
    scenario = scenarios.TrainInTimedOutOfDomainEvaluate(
        t, t_validation=t_validation, validation=True)
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
            t, t_validation=t, validation=True)

    t_validation = t - 10
    scenario = scenarios.TrainInTimedOutOfDomainWithLabelsEvaluate(
        t, t_validation=t_validation, validation=True)
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


@pytest.mark.parametrize("perc_users_train, perc_interactions_in",
                         [(0.7, 0.5), (0, 0.5)])
def test_strong_generalization_split(
        data_m, perc_users_train, perc_interactions_in):

    scenario = scenarios.StrongGeneralization(
        perc_users_train, perc_interactions_in)
    scenario.split(data_m)

    tr = scenario.training_data
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

    assert abs(len(tr_users) / (len(tr_users) + len(te_users)) -
               perc_users_train) < diff_allowed

    # Higher volatility, so not as bad to miss
    diff_allowed = 0.2
    assert abs(len(te_in_interactions) /
               (len(te_in_interactions) +
                len(te_out_interactions)) -
               perc_interactions_in) < diff_allowed

    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize("perc_users_train, perc_interactions_in",
                         [(0.7, 0.5), (0.3, 0.5)])
def test_strong_generalization_split_validation(
        data_m, perc_users_train, perc_interactions_in):

    # Filter a bit in the data_m, so we only have users with at least 2
    # interactions.
    events_per_user = data_m.binary_values.sum(axis=1)
    # Users with only a single interaction -> NAY
    events_per_user[events_per_user == 1] = 0
    users = set(events_per_user.nonzero()[0])
    data_m.users_in(list(users), inplace=True)

    scenario = scenarios.StrongGeneralization(
        perc_users_train, perc_interactions_in, validation=True)
    scenario.split(data_m)

    tr = scenario.training_data
    val_data_in, val_data_out = scenario.validation_data
    te_data_in, te_data_out = scenario.test_data

    tr_users = set(tr.indices[0])
    te_in_users = set(te_data_in.indices[0])
    te_out_users = set(te_data_out.indices[0])
    val_in_users = set(val_data_in.indices[0])
    val_out_users = set(val_data_out.indices[0])

    assert not tr_users.intersection(te_in_users)
    assert not tr_users.intersection(val_in_users)
    assert not te_in_users.intersection(val_in_users)

    assert te_in_users == te_out_users
    assert val_in_users == val_out_users

    # We expect the result to be approximately split, since it is random, it
    # is possible to not always be perfect.
    diff_allowed = 0.1
    tr_to_val_perc = len(tr_users) / (len(tr_users) + len(val_in_users))
    # this is a non configurable value
    assert abs(tr_to_val_perc - 0.8) < diff_allowed

    tr_and_val_to_te_perc = (len(tr_users) + len(val_in_users)) / \
        (len(tr_users) + len(val_in_users) + len(te_in_users))
    assert abs(tr_and_val_to_te_perc - perc_users_train) < diff_allowed

    assert val_data_out.active_users == val_data_in.active_users
    assert te_data_out.active_users == te_data_in.active_users
