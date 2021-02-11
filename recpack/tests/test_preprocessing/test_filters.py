import recpack.preprocessing.filters as filters
from recpack.data.matrix import InteractionMatrix


def test_min_users_per_item_filter(dataframe):

    df = dataframe

    # Not counting duplicates
    myfilter = filters.MinUsersPerItem(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
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
        InteractionMatrix.TIMESTAMP_IX,
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
        InteractionMatrix.TIMESTAMP_IX,
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
        InteractionMatrix.TIMESTAMP_IX,
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
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    # 2 and 5 have both 3 interactions, because 2 occurs a first time
    # before 5 in the dataframe, it gets selected.
    assert 5 not in filtered_iids


def test_nmost_recent(dataframe_with_fixed_timestamps):
    df = dataframe_with_fixed_timestamps

    myfilter = filters.NMostRecent(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
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
        "rating",
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
        min_rating=4,
    )
    filtered_df = myfilter.apply(df)

    assert filtered_df.shape == (4, 3)

    assert "rating" not in filtered_df.columns


def test_apply_all(dataframe):

    df = dataframe

    myfilter = filters.NMostPopular(
        3,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )
    filtered_df1, filtered_df2 = myfilter.apply_all(df, df.copy())

    filtered_iids = filtered_df1[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 not in filtered_iids

    filtered_iids = filtered_df2[InteractionMatrix.ITEM_IX].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 not in filtered_iids
