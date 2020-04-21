import pytest
import numpy as np

import recpack.splitters.scenarios as scenarios


@pytest.mark.parametrize("perc_users_in, t", [(0.7, 50), (0.5, 75), (0.3, 40)])
def test_strong_generalization_timed_split(data_m, perc_users_in, t):

    scenario = scenarios.StrongGeneralizationTimed(perc_users_in, t)
    scenario.split(data_m, data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    assert not set(tr.indices[0]).intersection(te_data_in.indices[0])
    assert not set(tr.indices[0]).intersection(te_data_out.indices[0])

    assert (tr.timestamps < t).all()
    assert (te_data_in.timestamps < t).all()
    assert (te_data_out.timestamps >= t).all()


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