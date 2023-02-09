# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import recpack.preprocessing.filters as filters
from recpack.matrix import InteractionMatrix


def test_min_users_per_item_filter(dataframe):

    df = dataframe

    # Not counting duplicates
    myfilter = filters.MinUsersPerItem(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
    )
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 not in filtered_iids

    # Counting duplicates
    myfilter = filters.MinUsersPerItem(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        count_duplicates=True,
    )
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 in filtered_iids


def test_min_items_per_user_filter(dataframe):

    df = dataframe

    # Not counting duplicates
    myfilter = filters.MinItemsPerUser(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
    )
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df[InteractionMatrix.USER_IX].unique()

    assert 0 in filtered_uids
    assert 1 in filtered_uids
    assert 2 in filtered_uids
    assert 3 not in filtered_uids
    assert 4 not in filtered_uids

    # Counting duplicates
    myfilter = filters.MinItemsPerUser(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        count_duplicates=True,
    )
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df[InteractionMatrix.USER_IX].unique()

    assert 0 in filtered_uids
    assert 1 in filtered_uids
    assert 2 in filtered_uids
    assert 3 not in filtered_uids
    assert 4 in filtered_uids


def test_nmost_popular(dataframe):

    df = dataframe

    myfilter = filters.NMostPopular(
        3,
        InteractionMatrix.ITEM_IX,
    )
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    # assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids

    # 2 and 5 have both 3 interactions, so one of the two should get selected.
    five_in = 5 in filtered_iids
    two_in = 2 in filtered_iids

    assert (five_in or two_in) and not (five_in and two_in)


def test_nmost_recent(dataframe_with_fixed_timestamps):
    df = dataframe_with_fixed_timestamps

    myfilter = filters.NMostRecent(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 not in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 in filtered_iids


def test_min_rating(rating_dataframe):

    df = rating_dataframe

    myfilter = filters.MinRating(
        4,
        "rating",
    )
    filtered_df = myfilter.apply(df)

    assert filtered_df.shape == (4, 3)

    assert "rating" not in filtered_df.columns


def test_deduplicate(dataframe_with_fixed_timestamps_inverted):
    df = dataframe_with_fixed_timestamps_inverted

    myfilter = filters.Deduplicate(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )

    filtered_df = myfilter.apply(df)

    # 4 duplicates
    assert filtered_df.shape[0] == dataframe_with_fixed_timestamps_inverted.shape[0] - 4
    # removes the last events
    assert filtered_df[filtered_df[InteractionMatrix.USER_IX] == 4][InteractionMatrix.TIMESTAMP_IX].max() == 1


def test_apply_all(dataframe):

    df = dataframe

    myfilter = filters.NMostPopular(
        3,
        InteractionMatrix.ITEM_IX,
    )
    filtered_df1, filtered_df2 = myfilter.apply_all(df, df.copy())

    filtered_iids = filtered_df1[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    # 2 and 5 have both 3 interactions, so one of the two should get selected.
    five_in = 5 in filtered_iids
    two_in = 2 in filtered_iids

    assert (five_in or two_in) and not (five_in and two_in)

    filtered_iids = filtered_df2[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    # 2 and 5 have both 3 interactions, so one of the two should get selected.
    five_in = 5 in filtered_iids
    two_in = 2 in filtered_iids

    assert (five_in or two_in) and not (five_in and two_in)


def test_max_items_per_user_filter(dataframe):

    df = dataframe

    # Not counting duplicates
    myfilter = filters.MaxItemsPerUser(
        2,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
    )
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df[InteractionMatrix.USER_IX].unique()

    assert 0 not in filtered_uids
    assert 1 not in filtered_uids
    assert 2 not in filtered_uids
    assert 3 in filtered_uids
    assert 4 in filtered_uids

    # Counting duplicates
    myfilter = filters.MaxItemsPerUser(
        2,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        count_duplicates=True,
    )
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df[InteractionMatrix.USER_IX].unique()

    assert 0 not in filtered_uids
    assert 1 not in filtered_uids
    assert 2 not in filtered_uids
    assert 3 in filtered_uids
    assert 4 not in filtered_uids
