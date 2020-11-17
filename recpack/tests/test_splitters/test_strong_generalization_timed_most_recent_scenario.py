import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios
from recpack.data.data_matrix import USER_IX, TIMESTAMP_IX


@pytest.mark.parametrize("t, n", [(99, 1), (99, 2), (99, -5)])
def test_strong_generalization_timed_split(data_m_w_dups, t, n):
    scenario = scenarios.StrongGeneralizationTimedMostRecent(t=t, n=n)
    scenario.split(data_m_w_dups)
    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    last_action_te_in = te_data_in.dataframe.groupby(USER_IX)[TIMESTAMP_IX].max()
    first_action_te_out = te_data_out.dataframe.groupby(USER_IX)[TIMESTAMP_IX].min()
    last_action_te_out = te_data_out.dataframe.groupby(USER_IX)[TIMESTAMP_IX].max()

    actions_per_user_total = data_m_w_dups.dataframe[USER_IX].value_counts()
    actions_per_user_tr = tr.dataframe[USER_IX].value_counts()
    actions_per_user_te_in = te_data_in.dataframe[USER_IX].value_counts()
    actions_per_user_te_out = te_data_out.dataframe[USER_IX].value_counts()
    actions_per_user_test = actions_per_user_total[actions_per_user_te_in.index]

    # User actions should never be split between train and test sets
    assert not tr.active_users.intersection(te_data_in.active_users)
    assert not tr.active_users.intersection(te_data_out.active_users)

    # No actions should ever be discarded, no set should be empty for this data
    assert (
        actions_per_user_tr.sum()
        + actions_per_user_te_in.sum()
        + actions_per_user_te_out.sum()
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
