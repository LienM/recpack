# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pandas as pd
import pytest
import numpy as np

from recpack.matrix import InteractionMatrix
import recpack.scenarios.splitters as splitters


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def data_m_w_timestamps():
    np.random.seed(42)

    num_users = 20
    num_items = 100
    num_interactions = 500

    min_t = 0
    max_t = 100

    input_dict = {
        InteractionMatrix.USER_IX: [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        InteractionMatrix.ITEM_IX: [np.random.randint(0, num_items) for _ in range(0, num_interactions)],
        InteractionMatrix.TIMESTAMP_IX: [np.random.randint(min_t, max_t) for _ in range(0, num_interactions)],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX], inplace=True)
    data = InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    return data


@pytest.fixture(scope="function")
def data_m_w_dups():
    np.random.seed(42)

    num_users = 20
    num_items = 100
    num_interactions = 500

    min_t = 0
    max_t = 100

    input_dict = {
        InteractionMatrix.USER_IX: [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        InteractionMatrix.ITEM_IX: [np.random.randint(0, num_items) for _ in range(0, num_interactions)],
        InteractionMatrix.TIMESTAMP_IX: [np.random.randint(min_t, max_t) for _ in range(0, num_interactions)],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(
        [
            InteractionMatrix.USER_IX,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.TIMESTAMP_IX,
        ],
        inplace=True,
    )
    data = InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    return data


def check_values_timestamps_match(data):
    indices = list(zip(*data.values.nonzero()))
    timestamps = data.timestamps

    pairs = timestamps.index
    for index_pair in indices:
        assert index_pair in pairs

    for index_pair in pairs:
        assert index_pair in indices

    assert data.values.sum() == data.timestamps.shape[0]


@pytest.mark.parametrize("in_perc", [0.45, 0.75, 0.25])
def test_strong_generalization_splitter(data_m_w_timestamps, in_perc):
    splitter = splitters.StrongGeneralizationSplitter(in_perc, seed=42, error_margin=0.10)

    tr, te = splitter.split(data_m_w_timestamps)

    real_perc = tr.values.sum() / data_m_w_timestamps.values.sum()

    assert np.isclose(real_perc, in_perc, atol=splitter.error_margin)

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("in_perc", [0.45, 0.75, 0.25])
def test_strong_generalization_splitter_w_dups(data_m_w_dups, in_perc):
    splitter = splitters.StrongGeneralizationSplitter(in_perc, seed=42, error_margin=0.10)

    tr, te = splitter.split(data_m_w_dups)

    real_perc = tr.values.sum() / data_m_w_dups.values.sum()

    assert np.isclose(real_perc, in_perc, atol=splitter.error_margin)

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize(
    "t, n_tr_expected, n_te_expected",
    [
        (2, 0, 3),
        (4, 1, 2),
        (5, 2, 1),
        (8, 3, 0),
    ],
)
def test_user_interaction_time_splitter(data_m_sessions, t, n_tr_expected, n_te_expected):
    splitter = splitters.UserInteractionTimeSplitter(t)

    tr, te = splitter.split(data_m_sessions)

    # No users are ever discarded
    assert tr.num_active_users + te.num_active_users == data_m_sessions.num_active_users

    # Users can have interactions in only one of the sets, never both
    assert not tr.active_users.intersection(te.active_users)

    # Each set contains every interaction of the users in it
    assert (tr._df[USER_IX].value_counts() == 3).all()
    assert (te._df[USER_IX].value_counts() == 3).all()

    # The 'on' parameter controls which user interaction is used to split on
    assert tr.num_active_users == n_tr_expected
    assert te.num_active_users == n_te_expected


@pytest.mark.parametrize("t", [20, 15])
def test_timestamp_splitter_no_limit(data_m_w_timestamps, t):
    splitter = splitters.TimestampSplitter(t)

    tr, te = splitter.split(data_m_w_timestamps)

    assert (tr.timestamps < t).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t", [20, 15])
def test_timestamp_splitter_no_limit_w_dups(data_m_w_dups, t):
    splitter = splitters.TimestampSplitter(t)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, delta_out", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_delta_out(data_m_w_timestamps, t, delta_out):
    splitter = splitters.TimestampSplitter(t, delta_out=delta_out)

    tr, te = splitter.split(data_m_w_timestamps)

    assert (tr.timestamps < t).all()

    assert (te.timestamps <= t + delta_out).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, delta_out", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_delta_out_w_dups(data_m_w_dups, t, delta_out):
    splitter = splitters.TimestampSplitter(t, delta_out=delta_out)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()

    assert (te.timestamps <= t + delta_out).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, delta_in", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_alpha(data_m_w_timestamps, t, delta_in):
    splitter = splitters.TimestampSplitter(t, delta_in=delta_in)

    tr, te = splitter.split(data_m_w_timestamps)

    assert (tr.timestamps < t).all()
    assert (tr.timestamps >= t - delta_in).all()

    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, delta_in", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_alpha_w_dups(data_m_w_dups, t, delta_in):
    splitter = splitters.TimestampSplitter(t, delta_in=delta_in)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()
    assert (tr.timestamps >= t - delta_in).all()

    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("n", [1, 2, -1, -2])
def test_most_recent_splitter(data_m_w_dups, n):
    m = data_m_w_dups
    last_action = m._df.groupby(USER_IX)[TIMESTAMP_IX].max()
    num_actions = m.values.toarray().sum(axis=1, keepdims=False)

    splitter = splitters.MostRecentSplitter(n)
    tr, te = splitter.split(m)

    # All users should have actions in both train and test sets
    assert tr.num_active_users == te.num_active_users == m.num_active_users

    for uid in tr.active_users:
        u_actions_tr = tr._df[tr._df[USER_IX] == uid]
        u_actions_te = te._df[te._df[USER_IX] == uid]
        # Train should contain all but n of a user's actions, test n actions

        # TODO This test breaks because of the duplicates... indices_in can't fix that.

        if n >= 0:
            assert len(u_actions_te) == n
            assert len(u_actions_tr) == num_actions[uid] - n
        # If n is negative, train contains |n| actions, test all but |n| actions
        else:
            assert len(u_actions_tr) == -n
            assert len(u_actions_te) == num_actions[uid] - (-n)
        # The most recent actions should be in the test set
        assert u_actions_te[TIMESTAMP_IX].max() == last_action[uid]


def test_user_splitter(data_m_w_timestamps):

    users = list(range(0, data_m_w_timestamps.shape[0]))

    np.random.shuffle(users)

    ix = data_m_w_timestamps.shape[0] // 2

    tr_u_in = users[:ix]
    te_u_in = users[ix:]

    splitter = splitters.UserSplitter(tr_u_in, te_u_in)
    tr, te = splitter.split(data_m_w_timestamps)

    tr_U, _ = tr.values.nonzero()
    te_U, _ = te.values.nonzero()

    assert not set(tr_U).difference(users[:ix])
    assert not set(te_U).difference(users[ix:])

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


# def test_user_splitter_no_full_split(data_m_w_timestamps):
#     splitter = splitter = splitter_base.UserSplitter([0, 1], [3])
#     with pytest.raises(AssertionError):
#         tr, te = splitter.split(data_m_w_timestamps)


@pytest.mark.parametrize("tr_perc", [0.75, 0.5, 0.45])
def test_perc_interaction_splitter(data_m_w_timestamps, tr_perc):
    num_interactions = len(data_m_w_timestamps.values.nonzero()[0])

    history_length = data_m_w_timestamps.values.sum(1)

    num_tr_interactions = np.ceil(history_length * tr_perc).sum()

    num_te_interactions = num_interactions - num_tr_interactions

    splitter = splitters.FractionInteractionSplitter(tr_perc, seed=42)
    tr, te = splitter.split(data_m_w_timestamps)

    assert len(tr.values.nonzero()[0]) == num_tr_interactions
    assert len(te.values.nonzero()[0]) == num_te_interactions

    assert tr.timestamps.shape[0] == num_tr_interactions
    assert te.timestamps.shape[0] == num_te_interactions
