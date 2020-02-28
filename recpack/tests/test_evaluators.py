import recpack.splits
import recpack.helpers
import recpack.evaluate

import pandas as pd
import pytest
import numpy


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {'userId': [1, 1, 1, 0, 0, 0], 'movieId': [1, 3, 4, 0, 2, 4], 'timestamp': [15, 26, 29, 10, 22, 34]}

    df = pd.DataFrame.from_dict(input_dict)
    data = recpack.helpers.create_data_M_from_pandas_df(df, 'movieId', 'userId', 'timestamp')
    return data


def test_fold_in_evaluator():
    data = generate_data()

    splitter = recpack.splits.TimedSplit(20, None)
    evaluator = recpack.evaluate.FoldInPercentage(0.5, seed=42)

    tr, val, te = splitter.split(data)
    in_, out_ = evaluator.split(tr, val, te)

    assert (in_.toarray() == numpy.array([
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0.]
    ])).all()
    assert (out_.toarray() == numpy.array([
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1.]
    ])).all()


def test_train_in_test_out_evaluator():
    data = generate_data()

    splitter = recpack.splits.TimedSplit(20, None)
    evaluator = recpack.evaluate.TrainingInTestOutEvaluator()

    tr, val, te = splitter.split(data)
    in_, out_ = evaluator.split(tr, val, te)

    assert (in_.toarray() == numpy.array([
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.]
    ])).all()
    assert (out_.toarray() == numpy.array([
        [0., 0., 1., 0., 1.],
        [0., 0., 0., 1., 1.]
    ])).all()
