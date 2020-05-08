import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split(data_m, perc_users_in, t):

    scenario = scenarios.StrongGeneralizationTimed(perc_users_in, t)
    tr, te_data_in, te_data_out = scenario.split(data_m)

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split(data_m, t):

    scenario = scenarios.TimedOutOfDomainPredictAndEvaluate(t)
    tr, te_data_in, te_data_out = scenario.split(data_m, data_m)

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_evaluate(data_m, t):

    scenario = scenarios.TrainInTimedOutOfDomainEvaluate(t)
    tr, te_data_in, te_data_out = scenario.split(data_m, data_m)

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()

    assert (tr.values.nonzero()[0] == te_data_in.values.nonzero()[0]).all()
    assert (tr.values.nonzero()[1] == te_data_in.values.nonzero()[1]).all()


@pytest.mark.parametrize("perc_users_train, perc_interactions_in", [(0.7, 0.5), (0, 0.5), (0.3, 0)])
def test_strong_generalization_split(data_m, perc_users_train, perc_interactions_in):

    scenario = scenarios.StrongGeneralization(perc_users_train, perc_interactions_in)
    tr, te_data_in, te_data_out =  scenario.split(data_m)

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    tr_users = set(tr.indices[0])
    te_in_users = set(te_data_in.indices[0])
    te_out_users = set(te_data_out.indices[0])
    te_users = te_in_users.union(te_out_users)

    te_in_interactions = te_data_in.indices[1]
    te_out_interactions = te_data_out.indices[1]

    # We expect the result to be approximately split, since it is random, it is possible to not always be perfect.
    diff_allowed = 0.1

    assert abs(len(tr_users) / (len(tr_users) + len(te_users)) - perc_users_train) < diff_allowed

    # Higher volatility, so not as bad to miss
    diff_allowed = 0.2
    assert abs(len(te_in_interactions) / (len(te_in_interactions) + len(te_out_interactions)) - perc_interactions_in) < diff_allowed
