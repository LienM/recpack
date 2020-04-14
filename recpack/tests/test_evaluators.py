import recpack.splits
from recpack.data_matrix import DataM
import recpack.evaluate
import scipy.sparse
import itertools

import pandas as pd
import pytest
import numpy


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {'userId': [1, 1, 1, 0, 0, 0], 'movieId': [1, 3, 4, 0, 2, 4], 'timestamp': [15, 26, 29, 10, 22, 34]}

    df = pd.DataFrame.from_dict(input_dict)
    data = DataM.create_from_dataframe(df, 'movieId', 'userId', 'timestamp')
    return data


def test_fold_in_evaluator():
    data = generate_data()

    splitter = recpack.splits.TimedSplit(20, None)
    evaluator = recpack.evaluate.FoldInPercentageEvaluator(0.5, seed=42)

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


def test_timed_test_split():
    data = generate_data()
    splitter = recpack.splits.TimedSplit(20, None)
    evaluator = recpack.evaluate.TimedSplitEvaluator(30)

    tr, val, te = splitter.split(data)
    in_, out_ = evaluator.split(tr, val, te)

    assert (in_.toarray() == numpy.array([
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 1.]
    ])).all()
    assert (out_.toarray() == numpy.array([
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.]
    ])).all()


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        3
    ]
)
def test_iterator(batch_size):
    class FoldInstance():
        def __init__(self, length=10, batch_size=1):
            self.len = length
            self.batch_size = batch_size
            assert self.len % 2 == 0
            self.sp_mat_in = scipy.sparse.diags([1 for i in range(self.len)]).tocsr()
            self.sp_mat_out = scipy.sparse.diags(
                list(itertools.chain.from_iterable([[1, 0] for i in range(int(self.len / 2))]))
            ).tocsr()

        def __iter__(self):
            return recpack.evaluate.evaluate.FoldIterator(self, self.batch_size)

    LEN = 10
    fold_i = FoldInstance(length=LEN, batch_size=batch_size)
    row_counts = []
    for in_, out_, users in fold_i:
        assert in_.nnz > 0
        assert out_.nnz > 0

        assert len(users) == in_.shape[0]
        assert len(users) == out_.shape[0]

        row_counts.append(in_.shape[0])

    # Because of how the out matrix is build, half of the rows should be skipped
    assert sum(row_counts) == (LEN / 2)
