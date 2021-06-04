import numpy as np

from recpack.splitters.scenarios import NextItemPrediction


def test_next_item_prediction_split(data_m_small):
    im = data_m_small
    scenario = NextItemPrediction(validation=False)
    scenario._split(im)

    train = scenario.train_X
    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out
    # Intersection asserts: every user should appear in data_out and data_in
    train_users = set(train.indices[0])
    test_data_out_users = set(test_data_out.indices[0])
    test_data_in_users = set(test_data_in.indices[0])
    assert train_users.intersection(test_data_in_users)
    assert train_users.intersection(test_data_out_users)
    # Length asserts: every user should have exactly one data_out and data_in entry
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(test_data_in.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)


def test_next_item_prediction_split_w_validation(data_m_small):
    im = data_m_small
    scenario = NextItemPrediction(validation=True)
    scenario._split(im)

    train = scenario.train_X
    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out
    val_data_in = scenario._validation_data_in
    val_data_out = scenario._validation_data_out
    # Intersection asserts: every user should appear in _out and _in
    train_users = set(train.indices[0])
    test_data_out_users = set(test_data_out.indices[0])
    test_data_in_users = set(test_data_in.indices[0])
    val_data_out_users = set(val_data_out.indices[0])
    val_data_in_users = set(val_data_in.indices[0])
    assert train_users.intersection(test_data_in_users)
    assert train_users.intersection(test_data_out_users)
    assert train_users.intersection(val_data_in_users)
    assert train_users.intersection(val_data_out_users)
    # Length asserts: every user should have exactly one _out and _in entry
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()
    assert val_data_out.shape[0] == im._df["uid"].nunique()
    assert val_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(test_data_in.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(val_data_in.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(val_data_out.values.sum(axis=1), 1)
