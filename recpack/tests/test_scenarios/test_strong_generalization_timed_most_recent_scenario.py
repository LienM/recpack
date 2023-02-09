# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import warnings

import recpack.scenarios as scenarios
from recpack.matrix import InteractionMatrix

USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.mark.parametrize("t, n", [(4, 1), (5, 1)])
def test_strong_generalization_timed_most_recent_split(data_m_sessions, t, n):
    scenario = scenarios.StrongGeneralizationTimedMostRecent(t=t, n_most_recent_out=n)
    scenario.split(data_m_sessions)
    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    # User earliest/latest interaction times, indexed by user
    last_action_te_in = te_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].max()
    first_action_te_out = te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].min()
    last_action_te_out = te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].max()

    # Nr. of user actions, indexed by user
    actions_per_user_total = data_m_sessions._df[USER_IX].value_counts()
    actions_per_user_tr = tr._df[USER_IX].value_counts()
    actions_per_user_te_in = te_data_in._df[USER_IX].value_counts()
    actions_per_user_te_out = te_data_out._df[USER_IX].value_counts()
    actions_per_user_test = actions_per_user_total[actions_per_user_te_in.index]

    # User actions should never be split between train and test sets
    assert not tr.active_users.intersection(te_data_in.active_users)
    assert not tr.active_users.intersection(te_data_out.active_users)

    # No actions should ever be discarded, no set should be empty for these settings
    assert (
        actions_per_user_tr.sum() + actions_per_user_te_in.sum() + actions_per_user_te_out.sum()
    ) == actions_per_user_total.sum()
    assert actions_per_user_tr.sum() > 0
    assert actions_per_user_te_in.sum() > 0
    assert actions_per_user_te_out.sum() > 0

    # Time of last user action decides if actions are placed in train or test
    assert (tr.timestamps < t).all()
    assert (last_action_te_out >= t).all()

    # All actions in the test_in set should occur before test_out actions
    assert (last_action_te_in <= first_action_te_out).all()

    # Correct amount of actions should be split off per user
    if n >= 0:
        assert (actions_per_user_test - actions_per_user_te_in == n).all()
        assert (actions_per_user_te_out == n).all()
    else:
        assert (actions_per_user_te_in == -n).all()
        assert (actions_per_user_test - actions_per_user_te_out == -n).all()


@pytest.mark.parametrize("t, t_val, n", [(5, 4, 1)])
def test_strong_generalization_timed_most_recent_w_val(data_m_sessions, t, t_val, n):
    scenario = scenarios.StrongGeneralizationTimedMostRecent(t=t, t_validation=t_val, n_most_recent_out=n, validation=True)
    scenario.split(data_m_sessions)
    val_tr = scenario.validation_training_data
    full_tr = scenario.full_training_data
    val_data_in, val_data_out = scenario.validation_data
    te_data_in, te_data_out = scenario.test_data

    # User earliest/latest interaction times, indexed by user
    last_action_val_in = val_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].max()
    first_action_val_out = val_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].min()
    last_action_val_out = val_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].max()
    last_action_te_in = te_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].max()
    first_action_te_out = te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].min()
    last_action_te_out = te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].max()

    # Nr. of user actions, indexed by user
    actions_per_user_total = data_m_sessions._df[USER_IX].value_counts()
    actions_per_user_tr = val_tr._df[USER_IX].value_counts()
    actions_per_user_val_in = val_data_in._df[USER_IX].value_counts()
    actions_per_user_val_out = val_data_out._df[USER_IX].value_counts()
    actions_per_user_val = actions_per_user_total[actions_per_user_val_in.index]
    actions_per_user_te_in = te_data_in._df[USER_IX].value_counts()
    actions_per_user_te_out = te_data_out._df[USER_IX].value_counts()
    actions_per_user_test = actions_per_user_total[actions_per_user_te_in.index]

    # User actions should never be split between train and validation or test sets
    assert not val_tr.active_users.intersection(val_data_in.active_users)
    assert not val_tr.active_users.intersection(val_data_out.active_users)
    assert not val_tr.active_users.intersection(te_data_in.active_users)
    assert not val_tr.active_users.intersection(te_data_out.active_users)

    # full training data should contain users from both validation_training
    # and validation evaluation datasets.
    assert full_tr.active_users.intersection(val_data_in.active_users) == val_data_in.active_users
    assert full_tr.active_users.intersection(val_tr.active_users) == val_tr.active_users

    # No actions should ever be discarded, no set should be empty for these settings
    assert (
        actions_per_user_tr.sum()
        + actions_per_user_val_in.sum()
        + actions_per_user_val_out.sum()
        + actions_per_user_te_in.sum()
        + actions_per_user_te_out.sum()
    ) == actions_per_user_total.sum()
    assert actions_per_user_tr.sum() > 0
    assert actions_per_user_val_in.sum() > 0
    assert actions_per_user_val_out.sum() > 0
    assert actions_per_user_te_in.sum() > 0
    assert actions_per_user_te_out.sum() > 0

    # Time of last user action decides if actions are placed in train or validation or test
    assert (val_tr.timestamps < t_val).all()
    assert (last_action_val_out >= t_val).all()
    assert (last_action_val_out < t).all()
    assert (last_action_te_out >= t).all()

    # All actions in the val_in set should occur before val_out actions
    assert (last_action_val_in <= first_action_val_out).all()

    # All actions in the test_in set should occur before test_out actions
    assert (last_action_te_in <= first_action_te_out).all()

    # Correct amount of actions should be split off per user
    if n >= 0:
        assert (actions_per_user_val - actions_per_user_val_in == n).all()
        assert (actions_per_user_val_out == n).all()
        assert (actions_per_user_test - actions_per_user_te_in == n).all()
        assert (actions_per_user_te_out == n).all()
    else:
        assert (actions_per_user_val_in == -n).all()
        assert (actions_per_user_val - actions_per_user_val_out == -n).all()
        assert (actions_per_user_te_in == -n).all()
        assert (actions_per_user_test - actions_per_user_te_out == -n).all()


@pytest.mark.parametrize("n", [0, -1, -4])
def incorrect_n_most_recent_out(n):
    with pytest.raises(ValueError) as e:
        scenarios.StrongGeneralizationTimedMostRecent(42, n_most_recent_out=n)

    assert e.match("strictly positive integer")


@pytest.mark.parametrize("t, n", [(4, 3), (4, 5)])
def test_strong_generalization_timed_most_recent_too_few_actions(data_m_sessions, t, n):
    scenario = scenarios.StrongGeneralizationTimedMostRecent(t=t, n_most_recent_out=n)

    # Splitting should warn user that one of the sets is empty
    with warnings.catch_warnings(record=True) as w:
        scenario.split(data_m_sessions)
        assert len(w) > 0

    te_data_in, te_data_out = scenario.test_data_in, scenario.test_data_out

    # Nr. of user actions, indexed by user
    actions_per_user_te_in = te_data_in._df[USER_IX].value_counts()
    actions_per_user_te_out = te_data_out._df[USER_IX].value_counts()

    # If n is positive and a user has <n actions, all are put in test_out
    if n >= 0:
        assert actions_per_user_te_in.sum() == 0
        assert (actions_per_user_te_out == 3).all()
    # If n is negative and a user has <|n| actions, all are put in test_in
    else:
        assert (actions_per_user_te_in == 3).all()
        assert actions_per_user_te_out.sum() == 0
