import pandas as pd
import pytest
import numpy as np

from recpack.data.data_matrix import DataM
import recpack.splitters.splitter_base as splitter_base


@pytest.fixture(scope="function")
def data_m_w_timestamps():
    np.random.seed(42)

    num_users = 20
    num_items = 100
    num_interactions = 500

    min_t = 0
    max_t = 100

    input_dict = {
        "userId": [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        "movieId": [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        "timestamp": [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(["userId", "movieId"], inplace=True)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", timestamp_ix="timestamp"
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
        "userId": [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        "movieId": [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        "timestamp": [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(["userId", "movieId", "timestamp"], inplace=True)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", timestamp_ix="timestamp"
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
    splitter = splitter_base.StrongGeneralizationSplitter(
        in_perc, seed=42, error_margin=0.10
    )

    tr, te = splitter.split(data_m_w_timestamps)

    real_perc = tr.values.sum() / data_m_w_timestamps.values.sum()

    assert np.isclose(real_perc, in_perc, atol=splitter.error_margin)

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("in_perc", [0.45, 0.75, 0.25])
def test_strong_generalization_splitter_w_dups(data_m_w_dups, in_perc):
    splitter = splitter_base.StrongGeneralizationSplitter(
        in_perc, seed=42, error_margin=0.10
    )

    tr, te = splitter.split(data_m_w_dups)

    real_perc = tr.values.sum() / data_m_w_dups.values.sum()

    assert np.isclose(real_perc, in_perc, atol=splitter.error_margin)

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t", [20, 15])
def test_timestamp_splitter_no_limit(data_m_w_timestamps, t):
    splitter = splitter_base.TimestampSplitter(t)

    tr, te = splitter.split(data_m_w_timestamps)

    assert (tr.timestamps < t).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t", [20, 15])
def test_timestamp_splitter_no_limit_w_dups(data_m_w_dups, t):
    splitter = splitter_base.TimestampSplitter(t)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_delta", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_t_delta(data_m_w_timestamps, t, t_delta):
    splitter = splitter_base.TimestampSplitter(t, t_delta=t_delta)

    tr, te = splitter.split(data_m_w_timestamps)

    assert (tr.timestamps < t).all()

    assert (te.timestamps <= t + t_delta).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_delta", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_t_delta_w_dups(data_m_w_dups, t, t_delta):
    splitter = splitter_base.TimestampSplitter(t, t_delta=t_delta)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()

    assert (te.timestamps <= t + t_delta).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_alpha", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_alpha(data_m_w_timestamps, t, t_alpha):
    splitter = splitter_base.TimestampSplitter(t, t_alpha=t_alpha)

    tr, te = splitter.split(data_m_w_timestamps)

    assert (tr.timestamps < t).all()
    assert (tr.timestamps >= t - t_alpha).all()

    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_alpha", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_alpha_w_dups(data_m_w_dups, t, t_alpha):
    splitter = splitter_base.TimestampSplitter(t, t_alpha=t_alpha)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()
    assert (tr.timestamps >= t - t_alpha).all()

    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


def test_user_splitter(data_m_w_timestamps):

    users = list(range(0, data_m_w_timestamps.shape[0]))

    np.random.shuffle(users)

    ix = data_m_w_timestamps.shape[0] // 2

    tr_u_in = users[:ix]
    te_u_in = users[ix:]

    splitter = splitter_base.UserSplitter(tr_u_in, te_u_in)
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

    splitter = splitter_base.PercentageInteractionSplitter(tr_perc, seed=42)
    tr, te = splitter.split(data_m_w_timestamps)

    assert len(tr.values.nonzero()[0]) == num_tr_interactions
    assert len(te.values.nonzero()[0]) == num_te_interactions

    assert tr.timestamps.shape[0] == num_tr_interactions
    assert te.timestamps.shape[0] == num_te_interactions


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_fold_iterator_correctness(data_m_w_timestamps, batch_size):
    splitter = splitter_base.PercentageInteractionSplitter(0.7, seed=42)

    data_m_in, data_m_out = splitter.split(data_m_w_timestamps)

    fold_iterator = splitter_base.FoldIterator(
        data_m_in, data_m_out, batch_size=batch_size
    )

    for fold_in, fold_out, users in fold_iterator:
        assert fold_in.nnz > 0

        assert fold_in.shape[0] == len(users)
        assert fold_out.shape[0] == len(users)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_fold_iterator_completeness(data_m_w_timestamps, batch_size):

    fold_iterator = splitter_base.FoldIterator(
        data_m_w_timestamps, data_m_w_timestamps, batch_size=batch_size)

    nonzero_users = set(data_m_w_timestamps.indices[0])

    all_batches = set()

    for fold_in, fold_out, users in fold_iterator:
        assert (fold_in != fold_out).nnz == 0

        users_in_batch = set(users)

        all_batches = all_batches.union(users_in_batch)

        assert len(users_in_batch) == batch_size or (
            len(users_in_batch) == len(nonzero_users) % batch_size
        )

    assert len(all_batches) == len(nonzero_users)
