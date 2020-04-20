import recpack.splits
from recpack.data_matrix import DataM
import math
import pandas as pd
import pytest
import math
import numpy


@pytest.fixture(scope="function")
def df():
    input_dict = {
        "userId": [3, 2, 1, 0],
        "movieId": [1, 0, 1, 0],
        "timestamp": [15, 26, 29, 100],
    }

    df = pd.DataFrame.from_dict(input_dict)
    data = DataM.create_from_dataframe(df, "movieId", "userId", "timestamp")
    return data


def check_values_timestamps_match(indices, timestamps):
    cnt = 0
    pairs = timestamps.index
    for index_pair in indices:
        assert index_pair in pairs
        cnt += 1

    assert cnt == timestamps.shape[0]  # Equal amnt of entries


def test_strong_generalization_split_w_validation_set(df):

    splitter = recpack.splits.StrongGeneralizationSplit(0.5, 0.25, seed=42)

    tr, val, te = splitter.split(df)

    # Check values splits -> If values split is okay, so it timestamps split normally
    assert (
        tr.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    ).all()
    assert (
        val.values.toarray()
        == numpy.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        te.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    ).all()

    tr_pairs = zip(*tr.values.nonzero())
    check_values_timestamps_match(tr_pairs, tr.timestamps)

    val_pairs = zip(*val.values.nonzero())
    check_values_timestamps_match(val_pairs, val.timestamps)

    te_pairs = zip(*te.values.nonzero())
    check_values_timestamps_match(te_pairs, te.timestamps)


def test_strong_generalization_split_no_validation_set(df):
    splitter = recpack.splits.StrongGeneralizationSplit(0.5, 0, seed=42)

    tr, val, te = splitter.split(df)

    # Check values splits
    assert (
        tr.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    ).all()
    assert (
        val.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        te.values.toarray()
        == numpy.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    ).all()

    tr_pairs = zip(*tr.values.nonzero())
    check_values_timestamps_match(tr_pairs, tr.timestamps)

    val_pairs = zip(*val.values.nonzero())
    check_values_timestamps_match(val_pairs, val.timestamps)

    te_pairs = zip(*te.values.nonzero())
    check_values_timestamps_match(te_pairs, te.timestamps)


def test_timed_split_no_limit_test(df):
    splitter = recpack.splits.TimedSplit(20, None)

    tr, val, te = splitter.split(df)

    # Check values splits
    assert (
        tr.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    ).all()
    assert (
        val.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        te.values.toarray()
        == numpy.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    ).all()

    tr_pairs = zip(*tr.values.nonzero())
    check_values_timestamps_match(tr_pairs, tr.timestamps)

    val_pairs = zip(*val.values.nonzero())
    check_values_timestamps_match(val_pairs, val.timestamps)

    te_pairs = zip(*te.values.nonzero())
    check_values_timestamps_match(te_pairs, te.timestamps)


def test_timed_split_windowed(df):
    splitter = recpack.splits.TimedSplit(20, 10)

    tr, val, te = splitter.split(df)

    # Check values splits
    assert (
        tr.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    ).all()
    assert (
        val.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        te.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    ).all()

    tr_pairs = zip(*tr.values.nonzero())
    check_values_timestamps_match(tr_pairs, tr.timestamps)

    val_pairs = zip(*val.values.nonzero())
    check_values_timestamps_match(val_pairs, val.timestamps)

    te_pairs = zip(*te.values.nonzero())
    check_values_timestamps_match(te_pairs, te.timestamps)


@pytest.mark.parametrize("T, T_ALPHA", [(20, 10), (20, 3)])
def test_timed_split_windowed_alpha(df, T, T_ALPHA):
    splitter = recpack.splits.TimedSplit(T, t_alpha=T_ALPHA)

    tr, val, te = splitter.split(df)

    assert (tr.timestamps < T).all()

    assert (te.timestamps >= T - T_ALPHA).all()


def test_predefined_split(df):
    splitter = recpack.splits.PredefinedUserSplit([0, 1], [2], [3], "ordered")
    tr, val, te = splitter.split(df)

    # Check values splits
    assert (
        tr.values.toarray()
        == numpy.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        val.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        te.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    ).all()

    # Check timestamp splits.
    tr_pairs = zip(*tr.values.nonzero())
    check_values_timestamps_match(tr_pairs, tr.timestamps)

    val_pairs = zip(*val.values.nonzero())
    check_values_timestamps_match(val_pairs, val.timestamps)

    te_pairs = zip(*te.values.nonzero())
    check_values_timestamps_match(te_pairs, te.timestamps)


def test_predefined_split_no_validation(df):
    splitter = recpack.splits.PredefinedUserSplit([0, 1], [], [2, 3], "ordered")
    tr, val, te = splitter.split(df)

    # Check values splits
    assert (
        tr.values.toarray()
        == numpy.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        val.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        te.values.toarray()
        == numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ).all()

    # Check timestamp splits.
    tr_pairs = zip(*tr.values.nonzero())
    check_values_timestamps_match(tr_pairs, tr.timestamps)

    val_pairs = zip(*val.values.nonzero())
    check_values_timestamps_match(val_pairs, val.timestamps)

    te_pairs = zip(*te.values.nonzero())
    check_values_timestamps_match(te_pairs, te.timestamps)


def test_predefined_split_no_full_split(df):
    splitter = recpack.splits.PredefinedUserSplit([0, 1], [], [3], "ordered")
    with pytest.raises(AssertionError):
        tr, val, te = splitter.split(df)


@pytest.mark.parametrize("tr_perc, val_perc", [(0.75, 0), (0.5, 0.25,), (0.45, 0.20)])
def test_weak_generalization(df, tr_perc, val_perc):
    num_interactions = len(df.values.nonzero()[0])

    num_tr_interactions = math.ceil(num_interactions * tr_perc)
    num_val_interactions = math.ceil(num_interactions * val_perc)
    num_te_interactions = num_interactions - num_tr_interactions - num_val_interactions

    splitter = recpack.splits.WeakGeneralizationSplit(tr_perc, val_perc, seed=42)
    tr, val, te = splitter.split(df)

    assert len(tr.values.nonzero()[0]) == num_tr_interactions
    assert len(val.values.nonzero()[0]) == num_val_interactions
    assert len(te.values.nonzero()[0]) == num_te_interactions

    assert tr.timestamps.shape[0] == num_tr_interactions
    assert val.timestamps.shape[0] == num_val_interactions
    assert te.timestamps.shape[0] == num_te_interactions


@pytest.mark.parametrize("val_perc", [0.0, 0.25, 0.5, 1.0])
def test_separate_data_for_validation_and_test(df, val_perc):
    num_interactions = len(df.values.nonzero()[0])
    num_evaluation_interactions = len(df.values.nonzero()[0])

    num_tr_interactions = num_interactions
    num_val_interactions = math.ceil(num_evaluation_interactions * val_perc)
    num_te_interactions = num_evaluation_interactions - num_val_interactions

    splitter = recpack.splits.SeparateDataForValidationAndTestSplit(val_perc, seed=42)
    tr, val, te = splitter.split(df, df)

    assert len(tr.values.nonzero()[0]) == num_tr_interactions
    assert len(val.values.nonzero()[0]) == num_val_interactions
    assert len(te.values.nonzero()[0]) == num_te_interactions

    assert tr.timestamps.shape[0] == num_tr_interactions
    assert val.timestamps.shape[0] == num_val_interactions
    assert te.timestamps.shape[0] == num_te_interactions


@pytest.mark.parametrize(
    "t, t_delta, t_alpha",
    [(20, None, None), (20, 10, None), (20, None, 10), (20, 10, 10)],
)
def test_separate_data_for_validation_and_test_timed_split(df, t, t_delta, t_alpha):

    splitter = recpack.splits.SeparateDataForValidationAndTestTimedSplit(t, t_delta)
    tr, val, te = splitter.split(df, df)
    
    assert (tr.timestamps < t).all()

    if t_alpha is not None:
        assert (tr.timestamps >= t - t_alpha).all()

    assert (te.timestamps >= t).all()

    if t_delta is not None:
        assert (te.timestamps < t + t_delta).all()

    # Assert validation is empty
    assert val.values.nnz == 0


@pytest.mark.parametrize(
    "t, t_delta, t_alpha",
    [(20, None, None), (20, 10, None), (20, None, 10), (20, 10, 10),],
)
def test_strong_generalization_timed_split(t, t_delta, t_alpha):
    input_dict = {
        "userId": [2, 1, 0, 0],
        "movieId": [1, 0, 1, 0],
        "timestamp": [15, 26, 10, 100],
    }

    df = pd.DataFrame.from_dict(input_dict)
    data = DataM.create_from_dataframe(df, "movieId", "userId", "timestamp")

    splitter = recpack.splits.StrongGeneralizationTimedSplit(
        t, t_delta=t_delta, t_alpha=t_alpha
    )

    tr, val, te = splitter.split(data)

    assert val.values.nnz == 0

    train_users = set(tr.timestamps.index.get_level_values(0))
    test_users = set(te.timestamps.index.get_level_values(0))

    assert (tr.timestamps < t).all()

    if t_alpha is not None:
        assert (tr.timestamps >= t - t_alpha).all()

    assert train_users.intersection(test_users) == set()

    assert (te.timestamps >= t).all()

    if t_delta is not None:
        assert (te.timestamps < t + t_delta).all()
