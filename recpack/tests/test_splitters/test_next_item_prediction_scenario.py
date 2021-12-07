import numpy as np
import pytest

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

        assert (
            train.values[u].sum() == len(sorted_item_history) - 1
        )  # all except the last item


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
    assert train_users.difference(test_data_in_users) == set()
    assert train_users.difference(test_data_out_users) == set()
    assert train_users.difference(val_data_in_users) == set()
    assert train_users.difference(val_data_out_users) == set()
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
            train.values[u].sum() == len(sorted_item_history) - 2
        )  #  All except validation and test event


@pytest.mark.parametrize("n_most_recent", [1, 2, 3])
def test_next_item_prediction_split_w_n_most_recent(data_m_small, n_most_recent):
    im = data_m_small
    scenario = NextItemPrediction(validation=False, n_most_recent=n_most_recent)
    scenario._split(im)

    train = scenario.train_X
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

    np.testing.assert_array_almost_equal(
        test_data_in.values.sum(axis=1) <= n_most_recent, True
    )
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)

    # Check that the right item has been selected.
    for u, sorted_item_history in im.sorted_item_history:
        assert test_data_out.values[u, sorted_item_history[-1]] == 1
        assert test_data_in.values[u, sorted_item_history[-2]] == 1
        assert test_data_in.values[u].sum() == min(
            len(sorted_item_history) - 1, n_most_recent
        )

        assert (
            train.values[u].sum() == len(sorted_item_history) - 1
        )  # all except the last item


@pytest.mark.parametrize("n_most_recent", [1, 2, 3])
def test_next_item_prediction_split_w_validation_w_n_most_recent(
    data_m_small, n_most_recent
):
    im = data_m_small
    scenario = NextItemPrediction(validation=True, n_most_recent=n_most_recent)
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
    assert train_users.difference(test_data_in_users) == set()
    assert train_users.difference(test_data_out_users) == set()
    assert train_users.difference(val_data_in_users) == set()
    assert train_users.difference(val_data_out_users) == set()
    # Length asserts: every user should have exactly one _out and _in entry
    assert test_data_out.shape[0] == im._df["uid"].nunique()
    assert test_data_in.shape[0] == im._df["uid"].nunique()
    assert val_data_out.shape[0] == im._df["uid"].nunique()
    assert val_data_in.shape[0] == im._df["uid"].nunique()

    np.testing.assert_array_almost_equal(
        test_data_in.values.sum(axis=1) <= n_most_recent, True
    )
    np.testing.assert_array_almost_equal(test_data_out.values.sum(axis=1), 1)
    np.testing.assert_array_almost_equal(
        val_data_in.values.sum(axis=1) <= n_most_recent, True
    )
    np.testing.assert_array_almost_equal(val_data_out.values.sum(axis=1), 1)

    # Check that the right item has been selected.
    for u, sorted_item_history in im.sorted_item_history:
        assert test_data_out.values[u, sorted_item_history[-1]] == 1
        assert test_data_in.values[u, sorted_item_history[-2]] == 1
        assert val_data_out.values[u, sorted_item_history[-2]] == 1

        assert test_data_in.values[u].sum() == min(
            len(sorted_item_history) - 1, n_most_recent
        )

        assert val_data_in.values[u].sum() == min(
            len(sorted_item_history) - 2, n_most_recent
        )

        assert (
            train.values[u].sum() == len(sorted_item_history) - 2
        )  #  All except validation and test event
