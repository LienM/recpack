import recpack.splits
import recpack.helpers
import pandas as pd
import pytest
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

    tr, val, te = splitter.split(data, None)

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

    tr, val, te = splitter.split(data, None)

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

    tr, val, te = splitter.split(data, None)

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

    tr, val, te = splitter.split(data, None)

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
    tr, val, te = splitter.split(data, None)

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
    tr, val, te = splitter.split(data, None)

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
        tr, val, te = splitter.split(data, None)
