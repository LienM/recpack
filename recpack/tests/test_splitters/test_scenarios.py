import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split(data_m, perc_users_in, t):

    scenario = scenarios.StrongGeneralizationTimed(perc_users_in, t)
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_split(data_m, t):

    scenario = scenarios.TimedOutOfDomainPredictAndEvaluate(t)
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()


@pytest.mark.parametrize("t", [50, 75, 40])
def test_timed_out_of_domain_evaluate(data_m, t):

    scenario = scenarios.TrainInTimedOutOfDomainEvaluate(t)
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()

    assert (tr.values.nonzero()[0] == te_data_in.values.nonzero()[0]).all()
    assert (tr.values.nonzero()[1] == te_data_in.values.nonzero()[1]).all()
