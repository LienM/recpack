import recpack.splits
import recpack.helpers
import math
import pandas as pd
import pytest
import math
import numpy


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {'userId': [3, 2, 1, 0], 'movieId': [1, 0, 1, 0], 'timestamp': [15, 26, 29, 100]}

    df = pd.DataFrame.from_dict(input_dict)
    data = recpack.helpers.create_data_M_from_pandas_df(df, 'movieId', 'userId', 'timestamp')
    return data


def test_strong_generalization_split_w_validation_set():

    data = generate_data()
    splitter = recpack.splits.StrongGeneralization(0.5, 0.25, seed=42)

    tr, val, te = splitter.split(data)

    # Check values splits
    assert (tr.values.toarray() == numpy.array([
        [0., 0.],
        [0., 1.],
        [0., 0.],
        [0., 1.]
    ])).all()
    assert (val.values.toarray() == numpy.array([
        [1., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 0.]
    ])).all()

    # Check timestamp splits.
    assert (tr.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 29.],
        [0., 0.],
        [0., 15.]
    ])).all()
    assert (val.timestamps.toarray() == numpy.array([
        [100., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [26., 0.],
        [0., 0.]
    ])).all()


def test_strong_generalization_split_no_validation_set():
    data = generate_data()
    splitter = recpack.splits.StrongGeneralization(0.5, 0, seed=42)

    tr, val, te = splitter.split(data)

    # Check values splits
    assert (tr.values.toarray() == numpy.array([
        [0., 0.],
        [0., 1.],
        [0., 0.],
        [0., 1.]
    ])).all()
    assert (val.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.values.toarray() == numpy.array([
        [1., 0.],
        [0., 0.],
        [1., 0.],
        [0., 0.]
    ])).all()

    # Check timestamp splits.
    assert (tr.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 29.],
        [0., 0.],
        [0., 15.]
    ])).all()
    assert (val.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.timestamps.toarray() == numpy.array([
        [100., 0.],
        [0., 0.],
        [26., 0.],
        [0., 0.]
    ])).all()


def test_timed_split_no_limit_test():
    data = generate_data()
    splitter = recpack.splits.TimedSplit(20, None)

    tr, val, te = splitter.split(data)

    # Check values splits
    assert (tr.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 1.]
    ])).all()
    assert (val.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.values.toarray() == numpy.array([
        [1., 0.],
        [0., 1.],
        [1., 0.],
        [0., 0.]
    ])).all()

    # Check timestamp splits.
    assert (tr.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 15.]
    ])).all()
    assert (val.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.timestamps.toarray() == numpy.array([
        [100., 0.],
        [0., 29.],
        [26., 0.],
        [0., 0.]
    ])).all()


def test_timed_split_windowed():
    data = generate_data()
    splitter = recpack.splits.TimedSplit(20, 10)

    tr, val, te = splitter.split(data)

    # Check values splits
    assert (tr.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 1.]
    ])).all()
    assert (val.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.values.toarray() == numpy.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [0., 0.]
    ])).all()

    # Check timestamp splits.
    assert (tr.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 15.]
    ])).all()
    assert (val.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 29.],
        [26., 0.],
        [0., 0.]
    ])).all()


def test_predefined_split():
    data = generate_data()
    splitter = recpack.splits.PredefinedUserSplit([0, 1], [2], [3], 'ordered')
    tr, val, te = splitter.split(data)

    # Check values splits
    assert (tr.values.toarray() == numpy.array([
        [1., 0.],
        [0., 1.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (val.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 0.]
    ])).all()
    assert (te.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 1.]
    ])).all()

    # Check timestamp splits.
    assert (tr.timestamps.toarray() == numpy.array([
        [100., 0.],
        [0., 29.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (val.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [26., 0.],
        [0., 0.]
    ])).all()
    assert (te.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 15.]
    ])).all()


def test_predefined_split_no_validation():
    data = generate_data()
    splitter = recpack.splits.PredefinedUserSplit([0, 1], [], [2, 3], 'ordered')
    tr, val, te = splitter.split(data)

    # Check values splits
    assert (tr.values.toarray() == numpy.array([
        [1., 0.],
        [0., 1.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (val.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.values.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 1.]
    ])).all()

    # Check timestamp splits.
    assert (tr.timestamps.toarray() == numpy.array([
        [100., 0.],
        [0., 29.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (val.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])).all()
    assert (te.timestamps.toarray() == numpy.array([
        [0., 0.],
        [0., 0.],
        [26., 0.],
        [0., 15.]
    ])).all()


def test_predefined_split_no_full_split():
    data = generate_data()
    splitter = recpack.splits.PredefinedUserSplit([0, 1], [], [3], 'ordered')
    with pytest.raises(AssertionError):
        tr, val, te = splitter.split(data)


@pytest.mark.parametrize(
    "tr_perc, val_perc",
    [
        (
            0.75,
            0
        ),
        (
            0.5,
            0.25,
        ),
        (
            0.45,
            0.20
        )
    ]
)
def test_weak_generalization(tr_perc, val_perc):
    data = generate_data()
    num_interactions = len(data.values.nonzero()[0])

    num_tr_interactions = math.ceil(num_interactions * tr_perc)
    num_val_interactions = math.ceil(num_interactions * val_perc)
    num_te_interactions = num_interactions - num_tr_interactions - num_val_interactions

    splitter = recpack.splits.WeakGeneralization(tr_perc, val_perc, seed=42)
    tr, val, te = splitter.split(data)

    assert len(tr.values.nonzero()[0]) == num_tr_interactions
    assert len(val.values.nonzero()[0]) == num_val_interactions
    assert len(te.values.nonzero()[0]) == num_te_interactions

    assert len(tr.timestamps.nonzero()[0]) == num_tr_interactions
    assert len(val.timestamps.nonzero()[0]) == num_val_interactions
    assert len(te.timestamps.nonzero()[0]) == num_te_interactions


@pytest.mark.parametrize(
    "val_perc",
    [
        0., 0.25, 0.5, 1.
    ]
)
def test_separate_data_for_validation_and_test(val_perc):
    data = generate_data()
    evaluation_data = generate_data()

    num_interactions = len(data.values.nonzero()[0])
    num_evaluation_interactions = len(evaluation_data.values.nonzero()[0])

    num_tr_interactions = num_interactions
    num_val_interactions = math.ceil(num_evaluation_interactions * val_perc)
    num_te_interactions = num_evaluation_interactions - num_val_interactions

    splitter = recpack.splits.SeparateDataForValidationAndTestSplit(val_perc, seed=42)
    tr, val, te = splitter.split(data, evaluation_data)

    assert len(tr.values.nonzero()[0]) == num_tr_interactions
    assert len(val.values.nonzero()[0]) == num_val_interactions
    assert len(te.values.nonzero()[0]) == num_te_interactions

    assert len(tr.timestamps.nonzero()[0]) == num_tr_interactions
    assert len(val.timestamps.nonzero()[0]) == num_val_interactions
    assert len(te.timestamps.nonzero()[0]) == num_te_interactions


@pytest.mark.parametrize(
    "t, t_delta",
    [
        (20, None,),
        (20, 10,)
    ]
)
def test_separate_data_for_validation_and_test_timed_split(t, t_delta):
    data = generate_data()
    evaluation_data = generate_data()

    splitter = recpack.splits.SeparateDataForValidationAndTestTimedSplit(t, t_delta)
    tr, val, te = splitter.split(data, evaluation_data)

    # Assert all data in train has timestamp < t
    tr_indices = tr.timestamps.nonzero()
    for i, j in zip(tr_indices[0], tr_indices[1]):
        assert tr.timestamps[i, j] < t

    # Assert all data in test has timestamp in [t, t+t_delta]
    te_indices = te.timestamps.nonzero()
    for i, j in zip(te_indices[0], te_indices[1]):
        ts = te.timestamps[i, j]
        if t_delta is None:
            assert ts >= t
        else:
            assert t <= ts and ts < t + t_delta

    # Assert validation is empty
    assert val.values.nnz == 0
    assert val.timestamps.nnz == 0
