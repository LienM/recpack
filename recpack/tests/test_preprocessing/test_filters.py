import recpack.preprocessing.filters as filters


def test_min_users_per_item_filter(dataframe):

    df = dataframe

    # Not counting duplicates
    myfilter = filters.MinUsersPerItem(3, "iid", "uid", "timestamp")
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df["iid"].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 not in filtered_iids

    # Counting duplicates
    myfilter = filters.MinUsersPerItem(3, "iid", "uid", "timestamp", count_duplicates=True)
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df["iid"].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 in filtered_iids


def test_min_items_per_user_filter(dataframe):

    df = dataframe

    # Not counting duplicates
    myfilter = filters.MinItemsPerUser(3, "iid", "uid", "timestamp")
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df["uid"].unique()

    assert 0 in filtered_uids
    assert 1 in filtered_uids
    assert 2 in filtered_uids
    assert 3 not in filtered_uids
    assert 4 not in filtered_uids

    # Counting duplicates
    myfilter = filters.MinItemsPerUser(
        3, "iid", "uid", "timestamp", count_duplicates=True
    )
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df["uid"].unique()

    assert 0 in filtered_uids
    assert 1 in filtered_uids
    assert 2 in filtered_uids
    assert 3 not in filtered_uids
    assert 4 in filtered_uids


def test_nmost_popular(dataframe):

    df = dataframe

    myfilter = filters.NMostPopular(3, "iid", "uid", "timestamp")
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df["iid"].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 not in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids
    assert 5 in filtered_iids
