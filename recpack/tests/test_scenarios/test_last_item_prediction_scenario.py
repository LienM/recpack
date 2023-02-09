# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pytest

from recpack.scenarios import LastItemPrediction


def test_next_item_prediction_split(data_m_small):
    im = data_m_small
    scenario = LastItemPrediction(validation=False, n_most_recent_in=1)
    scenario._split(im)

    train = scenario.full_training_data
    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out
    # Intersection asserts: every user should appear in data_out and data_in
    train_users = set(train.indices[0])
    test_data_out_users = set(test_data_out.indices[0])
    test_data_in_users = set(test_data_in.indices[0])
    assert train_users.difference(test_data_in_users) == set()
    assert train_users.difference(test_data_out_users) == set()
    # Length asserts: every user should have exactly one data_out and data_in row
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(test_data_in.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)

    # Check that the right item has been selected.
    for u, sorted_item_history in im.sorted_item_history:
        assert test_data_out.values[u, sorted_item_history[-1]] == 1
        assert test_data_in.values[u, sorted_item_history[-2]] == 1

        assert train.values[u].sum() == len(sorted_item_history) - 1  # all except the last item


def test_next_item_prediction_split_w_validation(data_m_small):
    im = data_m_small
    scenario = LastItemPrediction(validation=True, n_most_recent_in=1)
    scenario._split(im)

    validation_training_data = scenario.validation_training_data
    full_training_data = scenario.full_training_data
    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out
    val_data_in = scenario._validation_data_in
    val_data_out = scenario._validation_data_out
    # Intersection asserts: every user should appear in _out and _in
    validation_train_users = set(validation_training_data.indices[0])
    full_train_users = set(full_training_data.indices[0])
    test_data_out_users = set(test_data_out.indices[0])
    test_data_in_users = set(test_data_in.indices[0])
    val_data_out_users = set(val_data_out.indices[0])
    val_data_in_users = set(val_data_in.indices[0])
    assert validation_train_users.difference(test_data_in_users) == set()
    assert validation_train_users.difference(test_data_out_users) == set()
    assert validation_train_users.difference(val_data_in_users) == set()
    assert validation_train_users.difference(val_data_out_users) == set()

    assert full_train_users.difference(test_data_in_users) == set()
    assert full_train_users.difference(test_data_out_users) == set()
    assert full_train_users.difference(val_data_in_users) == set()
    assert full_train_users.difference(val_data_out_users) == set()

    # Length asserts: every user should have exactly one _out and _in entry
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()
    assert val_data_out.shape[0] == im._df["uid"].nunique()
    assert val_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(test_data_in.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(val_data_in.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(val_data_out.values.sum(axis=1), 1)

    # Check that the right item has been selected.
    for u, sorted_item_history in im.sorted_item_history:
        assert test_data_out.values[u, sorted_item_history[-1]] == 1
        assert test_data_in.values[u, sorted_item_history[-2]] == 1
        assert val_data_out.values[u, sorted_item_history[-2]] == 1
        assert val_data_in.values[u, sorted_item_history[-3]] == 1

        assert (
            validation_training_data.values[u].sum() == len(sorted_item_history) - 2
        )  #  All except validation and test event

        assert full_training_data.values[u].sum() == len(sorted_item_history) - 1  #  All except test event


@pytest.mark.parametrize("n_most_recent_in", [1, 2, 3])
def test_next_item_prediction_split_w_n_most_recent_in(data_m_small, n_most_recent_in):
    im = data_m_small
    scenario = LastItemPrediction(validation=False, n_most_recent_in=n_most_recent_in)
    scenario._split(im)

    train = scenario.full_training_data
    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out
    # Intersection asserts: every user should appear in data_out and data_in
    train_users = set(train.indices[0])
    test_data_out_users = set(test_data_out.indices[0])
    test_data_in_users = set(test_data_in.indices[0])
    assert train_users.difference(test_data_in_users) == set()
    assert train_users.difference(test_data_out_users) == set()
    # Length asserts: every user should have exactly one data_out and data_in entry
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(test_data_in.values.sum(axis=1) <= n_most_recent_in, True)
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)

    # Check that the right item has been selected.
    for u, sorted_item_history in im.sorted_item_history:
        assert test_data_out.values[u, sorted_item_history[-1]] == 1
        assert test_data_in.values[u, sorted_item_history[-2]] == 1
        assert test_data_in.values[u].sum() == min(len(sorted_item_history) - 1, n_most_recent_in)

        assert train.values[u].sum() == len(sorted_item_history) - 1  # all except the last item


@pytest.mark.parametrize("n_most_recent_in", [1, 2, 3])
def test_next_item_prediction_split_w_validation_w_n_most_recent_in(data_m_small, n_most_recent_in):
    im = data_m_small
    scenario = LastItemPrediction(validation=True, n_most_recent_in=n_most_recent_in)
    scenario._split(im)

    full_training_data = scenario.full_training_data
    validation_training_data = scenario.validation_training_data
    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out
    val_data_in = scenario._validation_data_in
    val_data_out = scenario._validation_data_out
    # Intersection asserts: every user should appear in _out and _in
    validation_train_users = set(validation_training_data.indices[0])
    full_train_users = set(full_training_data.indices[0])
    test_data_out_users = set(test_data_out.indices[0])
    test_data_in_users = set(test_data_in.indices[0])
    val_data_out_users = set(val_data_out.indices[0])
    val_data_in_users = set(val_data_in.indices[0])
    assert validation_train_users.difference(test_data_in_users) == set()
    assert validation_train_users.difference(test_data_out_users) == set()
    assert validation_train_users.difference(val_data_in_users) == set()
    assert validation_train_users.difference(val_data_out_users) == set()

    assert full_train_users.difference(test_data_in_users) == set()
    assert full_train_users.difference(test_data_out_users) == set()
    assert full_train_users.difference(val_data_in_users) == set()
    assert full_train_users.difference(val_data_out_users) == set()

    # Length asserts: every user should have exactly one _out and _in entry
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()
    assert val_data_out.shape[0] == im._df["uid"].nunique()
    assert val_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(test_data_in.values.sum(axis=1) <= n_most_recent_in, True)
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(val_data_in.values.sum(axis=1) <= n_most_recent_in, True)
    np.testing.assert_array_almost_equal(val_data_out.values.sum(axis=1), 1)

    # Check that the right item has been selected.
    for u, sorted_item_history in im.sorted_item_history:
        assert test_data_out.values[u, sorted_item_history[-1]] == 1
        assert test_data_in.values[u, sorted_item_history[-2]] == 1
        assert val_data_out.values[u, sorted_item_history[-2]] == 1

        assert test_data_in.values[u].sum() == min(len(sorted_item_history) - 1, n_most_recent_in)

        assert val_data_in.values[u].sum() == min(len(sorted_item_history) - 2, n_most_recent_in)

        assert (
            validation_training_data.values[u].sum() == len(sorted_item_history) - 2
        )  #  All except validation and test event
        assert (
            full_training_data.values[u].sum() == len(sorted_item_history) - 1
        )  #  All except validation and test event


def test_next_item_prediction_split_all_history(data_m_small):
    im = data_m_small
    scenario = LastItemPrediction(validation=False)
    scenario._split(im)

    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out

    assert im.num_interactions == test_data_in.num_interactions + test_data_out.num_interactions
