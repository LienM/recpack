# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pandas as pd
import pytest

from recpack.scenarios import TimedLastItemPrediction
from recpack.tests.conftest import USER_IX, TIMESTAMP_IX


@pytest.mark.parametrize("t", [4, 5])
def test_timed_most_recent_split(data_m_sessions, t):
    scenario = TimedLastItemPrediction(t=t)
    scenario.split(data_m_sessions)
    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    # All users should have maximally 1 test data out point
    test_out_user_counts = (
        te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_out_user_counts").reset_index()
    )
    np.testing.assert_array_equal(test_out_user_counts.test_out_user_counts, 1)
    # Test datapoints should be beyond t
    assert te_data_out._df[TIMESTAMP_IX].min() >= t
    # Training data points should be before t
    assert tr._df[TIMESTAMP_IX].max() < t

    full_data_user_counts = (
        data_m_sessions._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("full_data_user_counts").reset_index()
    )

    test_in_user_counts = (
        te_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_in_user_counts").reset_index()
    )
    test_counts = pd.merge(test_in_user_counts, test_out_user_counts)
    test_counts = pd.merge(test_counts, full_data_user_counts, how="left")

    # The testing users should have been split with 1 item
    # on the right and the rest on the left.
    assert (
        test_counts["test_in_user_counts"] + test_counts["test_out_user_counts"]
        == test_counts["full_data_user_counts"]
    ).all()


@pytest.mark.parametrize("t, t_val", [(5, 4)])
def test_timed_most_recent_w_val(data_m_sessions, t, t_val):
    scenario = TimedLastItemPrediction(t=t, t_validation=t_val, validation=True)
    scenario.split(data_m_sessions)
    val_tr = scenario.validation_training_data
    full_tr = scenario.full_training_data
    val_data_in, val_data_out = scenario.validation_data
    te_data_in, te_data_out = scenario.test_data

    # All users should have maximally 1 test data out point
    test_out_user_counts = (
        te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_out_user_counts").reset_index()
    )
    np.testing.assert_array_equal(test_out_user_counts.test_out_user_counts, 1)
    # Test datapoints should be beyond t
    assert te_data_out._df[TIMESTAMP_IX].min() >= t
    # Training data points should be before t
    assert full_tr._df[TIMESTAMP_IX].max() < t
    # validation training data should be before t_val
    assert val_tr._df[TIMESTAMP_IX].max() < t_val

    # All users should have maximally 1 validation data out point
    val_out_user_counts = (
        val_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("val_out_user_counts").reset_index()
    )
    np.testing.assert_array_equal(val_out_user_counts.val_out_user_counts, 1)
    # validation datapoints should be beyond t_val and before t
    assert val_data_out._df[TIMESTAMP_IX].min() >= t_val
    assert val_data_out._df[TIMESTAMP_IX].max() < t

    full_data_user_counts = (
        data_m_sessions._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("full_data_user_counts").reset_index()
    )

    test_in_user_counts = (
        te_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_in_user_counts").reset_index()
    )
    test_counts = pd.merge(test_in_user_counts, test_out_user_counts)
    test_counts = pd.merge(test_counts, full_data_user_counts, how="left")

    # The testing users should have been split with 1 item
    # on the right and the rest on the left.
    assert (
        test_counts["test_in_user_counts"] + test_counts["test_out_user_counts"]
        == test_counts["full_data_user_counts"]
    ).all()

    full_val_data_user_counts = (
        full_tr._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("val_data_user_counts").reset_index()
    )
    val_in_user_counts = (
        val_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("val_in_user_counts").reset_index()
    )

    val_counts = pd.merge(val_in_user_counts, val_out_user_counts)
    val_counts = pd.merge(val_counts, full_val_data_user_counts, how="left")

    # The testing users should have been split with 1 item
    # on the right and the rest on the left.
    assert (
        val_counts["val_in_user_counts"] + val_counts["val_out_user_counts"] == val_counts["val_data_user_counts"]
    ).all()


@pytest.mark.parametrize("t, n_most_recent_in", [(4, 1), (5, 2)])
def test_timed_last_item_split_n_most_recent_in(data_m_sessions, t, n_most_recent_in):
    scenario = TimedLastItemPrediction(t=t, n_most_recent_in=n_most_recent_in)
    scenario.split(data_m_sessions)
    tr = scenario.full_training_data
    te_data_in, te_data_out = scenario.test_data

    # All users should have maximally 1 test data out point
    test_out_user_counts = (
        te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_out_user_counts").reset_index()
    )
    np.testing.assert_array_equal(test_out_user_counts.test_out_user_counts, 1)
    # Test datapoints should be beyond t
    assert te_data_out._df[TIMESTAMP_IX].min() >= t
    # Training data points should be before t
    assert tr._df[TIMESTAMP_IX].max() < t

    full_data_user_counts = (
        data_m_sessions._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("full_data_user_counts").reset_index()
    )

    test_in_user_counts = (
        te_data_in._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_in_user_counts").reset_index()
    )
    test_counts = pd.merge(test_in_user_counts, test_out_user_counts)
    test_counts = pd.merge(test_counts, full_data_user_counts, how="left")

    # The testing users should have been split with 1 item
    # on the right and the `n_most_recent_in` on the left.
    assert (test_counts["test_out_user_counts"] == 1).all()

    assert (test_counts["test_in_user_counts"] == n_most_recent_in).all()


@pytest.mark.parametrize("t, delta_out", [(4, 1), (5, 2)])
def test_timed_last_item_split_delta_out(data_m_sessions, t, delta_out):
    scenario = TimedLastItemPrediction(t=t, delta_out=delta_out)
    scenario.split(data_m_sessions)
    tr = scenario.full_training_data
    _, te_data_out = scenario.test_data

    # All users should have maximally 1 test data out point
    test_out_user_counts = (
        te_data_out._df.groupby(USER_IX)[TIMESTAMP_IX].count().rename("test_out_user_counts").reset_index()
    )
    np.testing.assert_array_equal(test_out_user_counts.test_out_user_counts, 1)
    # Test datapoints should be beyond t
    assert te_data_out._df[TIMESTAMP_IX].min() >= t
    # Test datapoints should be before t + delta_out
    assert te_data_out._df[TIMESTAMP_IX].max() <= (t + delta_out)
    # Training data points should be before t
    assert tr._df[TIMESTAMP_IX].max() < t


@pytest.mark.parametrize(
    "t, n_most_recent_in, expected_error",
    [
        (5, 0, "Using n_most_recent_in = 0 is not supported."),
    ],
)
def test_timed_last_item_split_zero_values_for_n(t, n_most_recent_in, expected_error):
    with pytest.raises(ValueError) as value_error:
        TimedLastItemPrediction(t=t, n_most_recent_in=n_most_recent_in)

    assert value_error.match(expected_error)


@pytest.mark.parametrize(
    "validation, t_validation",
    [
        (True, None),
    ],
)
def test_timed_last_item_split_n_most_recent_in_no_t_val(validation, t_validation):
    with pytest.raises(Exception) as error:
        TimedLastItemPrediction(t=2, n_most_recent_in=1, validation=validation, t_validation=t_validation)

    assert error.match("t_validation should be provided when using validation split.")
