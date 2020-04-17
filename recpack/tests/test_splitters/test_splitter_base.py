import recpack.splitters.splitter_base as splitters
from recpack.data_matrix import DataM
import math
import pandas as pd
import pytest
import numpy as np

num_users = 250
num_items = 100
num_interactions = 500

min_t = 0
max_t = 100


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
def test_strong_generalization_splitter(data_m, in_perc):
    splitter = splitters.StrongGeneralizationSplitter(in_perc, seed=42, error_margin=0.10)

    tr, te = splitter.split(data_m)

    real_perc = tr.values.sum() / data_m.values.sum()

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


@pytest.mark.parametrize("t", [20, 15])
def test_timestamp_splitter_no_limit(data_m, t):
    splitter = splitters.TimestampSplitter(t)

    tr, te = splitter.split(data_m)

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


@pytest.mark.parametrize("t, t_delta", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_t_delta(data_m, t, t_delta):
    splitter = splitters.TimestampSplitter(t, t_delta=t_delta)

    tr, te = splitter.split(data_m)

    assert (tr.timestamps < t).all()

    assert (te.timestamps <= t + t_delta).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_delta", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_t_delta_w_dups(data_m_w_dups, t, t_delta):
    splitter = splitters.TimestampSplitter(t, t_delta=t_delta)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()

    assert (te.timestamps <= t + t_delta).all()
    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_alpha", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_alpha(data_m, t, t_alpha):
    splitter = splitters.TimestampSplitter(t, t_alpha=t_alpha)

    tr, te = splitter.split(data_m)

    assert (tr.timestamps < t).all()
    assert (tr.timestamps >= t - t_alpha).all()

    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


@pytest.mark.parametrize("t, t_alpha", [(20, 10), (20, 3)])
def test_timestamp_splitter_windowed_alpha_w_dups(data_m_w_dups, t, t_alpha):
    splitter = splitters.TimestampSplitter(t, t_alpha=t_alpha)

    tr, te = splitter.split(data_m_w_dups)

    assert (tr.timestamps < t).all()
    assert (tr.timestamps >= t - t_alpha).all()

    assert (te.timestamps >= t).all()

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


def test_user_splitter(data_m):

    users = list(range(0, data_m.shape[0]))

    np.random.shuffle(users)

    ix = data_m.shape[0] // 2

    tr_u_in = users[:ix]
    te_u_in = users[ix:]

    splitter = splitters.UserSplitter(tr_u_in, te_u_in)
    tr, te = splitter.split(data_m)

    tr_U, _ = tr.values.nonzero()
    te_U, _ = te.values.nonzero()

    assert not set(tr_U).difference(users[:ix])
    assert not set(te_U).difference(users[ix:])

    check_values_timestamps_match(tr)
    check_values_timestamps_match(te)


def test_user_splitter_no_full_split(data_m):
    splitter = splitter = splitters.UserSplitter([0, 1], [3])
    with pytest.raises(AssertionError):
        tr, te = splitter.split(data_m)


# @pytest.mark.parametrize("tr_perc", [0.75, 0.5, 0.45])
# def test_weak_generalization(data_m, tr_perc):
#     num_interactions = len(data_m.values.nonzero()[0])

#     num_tr_interactions = math.ceil(num_interactions * tr_perc)
#     num_te_interactions = num_interactions - num_tr_interactions

#     splitter = splitters.PercentageInteractionSplitter(tr_perc, seed=42)
#     tr, te = splitter.split(data_m)

#     print(tr.values.toarray())

#     assert len(tr.values.nonzero()[0]) == num_tr_interactions
#     assert len(te.values.nonzero()[0]) == num_te_interactions

#     assert tr.timestamps.shape[0] == num_tr_interactions
#     assert te.timestamps.shape[0] == num_te_interactions


# @pytest.mark.parametrize("val_perc", [0.0, 0.25, 0.5, 1.0])
# def test_separate_data_for_validation_and_test(df, val_perc):
#     num_interactions = len(df.values.nonzero()[0])
#     num_evaluation_interactions = len(df.values.nonzero()[0])

#     num_tr_interactions = num_interactions
#     num_val_interactions = math.ceil(num_evaluation_interactions * val_perc)
#     num_te_interactions = num_evaluation_interactions - num_val_interactions

#     splitter = recpack.splits.SeparateDataForValidationAndTestSplit(val_perc, seed=42)
#     tr, val, te = splitter.split(df, df)

#     assert len(tr.values.nonzero()[0]) == num_tr_interactions
#     assert len(val.values.nonzero()[0]) == num_val_interactions
#     assert len(te.values.nonzero()[0]) == num_te_interactions

#     assert tr.timestamps.shape[0] == num_tr_interactions
#     assert val.timestamps.shape[0] == num_val_interactions
#     assert te.timestamps.shape[0] == num_te_interactions


# @pytest.mark.parametrize(
#     "t, t_delta, t_alpha",
#     [(20, None, None), (20, 10, None), (20, None, 10), (20, 10, 10)],
# )
# def test_separate_data_for_validation_and_test_timed_split(df, t, t_delta, t_alpha):

#     splitter = recpack.splits.SeparateDataForValidationAndTestTimedSplit(t, t_delta)
#     tr, val, te = splitter.split(df, df)

#     assert (tr.timestamps < t).all()

#     if t_alpha is not None:
#         assert (tr.timestamps >= t - t_alpha).all()

#     assert (te.timestamps >= t).all()

#     if t_delta is not None:
#         assert (te.timestamps < t + t_delta).all()

#     # Assert validation is empty
#     assert val.values.nnz == 0


# @pytest.mark.parametrize(
#     "t, t_delta, t_alpha",
#     [(20, None, None), (20, 10, None), (20, None, 10), (20, 10, 10),],
# )
# def test_strong_generalization_timed_split(t, t_delta, t_alpha):
#     input_dict = {
#         "userId": [2, 1, 0, 0],
#         "movieId": [1, 0, 1, 0],
#         "timestamp": [15, 26, 10, 100],
#     }

#     df = pd.DataFrame.from_dict(input_dict)
#     data = DataM.create_from_dataframe(df, "movieId", "userId", "timestamp")

#     splitter = recpack.splits.StrongGeneralizationTimedSplit(
#         t, t_delta=t_delta, t_alpha=t_alpha
#     )

#     tr, val, te = splitter.split(data)

#     assert val.values.nnz == 0

#     train_users = set(tr.timestamps.index.get_level_values(0))
#     test_users = set(te.timestamps.index.get_level_values(0))

#     assert (tr.timestamps < t).all()

#     if t_alpha is not None:
#         assert (tr.timestamps >= t - t_alpha).all()

#     assert train_users.intersection(test_users) == set()

#     assert (te.timestamps >= t).all()

#     if t_delta is not None:
#         assert (te.timestamps < t + t_delta).all()



# def generate_data():
#     # TODO move this test input to a conftest file as a fixture
#     input_dict = {'userId': [1, 1, 1, 0, 0, 0], 'movieId': [1, 3, 4, 0, 2, 4], 'timestamp': [15, 26, 29, 10, 22, 34]}

#     df = pd.DataFrame.from_dict(input_dict)
#     data = DataM.create_from_dataframe(df, 'movieId', 'userId', 'timestamp')
#     return data


# def test_fold_in_evaluator():
#     data = generate_data()

#     splitter = recpack.splits.TimedSplit(20, None)
#     evaluator = recpack.evaluate.FoldInPercentageEvaluator(0.5, seed=42)

#     tr, val, te = splitter.split(data)
#     in_, out_ = evaluator.split(tr, val, te)

#     assert (in_.toarray() == np.array([
#         [0., 0., 0., 0., 1.],
#         [0., 0., 0., 1., 0.]
#     ])).all()
#     assert (out_.toarray() == np.array([
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 0., 1.]
#     ])).all()


# def test_train_in_test_out_evaluator():
#     data = generate_data()

#     splitter = recpack.splits.TimedSplit(20, None)
#     evaluator = recpack.evaluate.TrainingInTestOutEvaluator()

#     tr, val, te = splitter.split(data)
#     in_, out_ = evaluator.split(tr, val, te)

#     assert (in_.toarray() == np.array([
#         [1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.]
#     ])).all()
#     assert (out_.toarray() == np.array([
#         [0., 0., 1., 0., 1.],
#         [0., 0., 0., 1., 1.]
#     ])).all()


# def test_timed_test_split():
#     data = generate_data()
#     splitter = recpack.splits.TimedSplit(20, None)
#     evaluator = recpack.evaluate.TimedSplitEvaluator(30)

#     tr, val, te = splitter.split(data)
#     in_, out_ = evaluator.split(tr, val, te)

#     assert (in_.toarray() == np.array([
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 1., 1.]
#     ])).all()
#     assert (out_.toarray() == np.array([
#         [0., 0., 0., 0., 1.],
#         [0., 0., 0., 0., 0.]
#     ])).all()


# @pytest.mark.parametrize(
#     "batch_size",
#     [
#         1,
#         2,
#         3
#     ]
# )
# def test_iterator(batch_size):
#     class FoldInstance():
#         def __init__(self, length=10, batch_size=1):
#             self.len = length
#             self.batch_size = batch_size
#             assert self.len % 2 == 0
#             self.sp_mat_in = scipy.sparse.diags([1 for i in range(self.len)]).tocsr()
#             self.sp_mat_out = scipy.sparse.diags(
#                 list(itertools.chain.from_iterable([[1, 0] for i in range(int(self.len / 2))]))
#             ).tocsr()

#         def __iter__(self):
#             return recpack.evaluate.evaluators.FoldIterator(self, self.batch_size)

#     LEN = 10
#     fold_i = FoldInstance(length=LEN, batch_size=batch_size)
#     row_counts = []
#     for in_, out_, users in fold_i:
#         assert in_.nnz > 0
#         assert out_.nnz > 0

#         assert len(users) == in_.shape[0]
#         assert len(users) == out_.shape[0]

#         row_counts.append(in_.shape[0])

#     # Because of how the out matrix is build, half of the rows should be skipped
#     assert sum(row_counts) == (LEN / 2)
