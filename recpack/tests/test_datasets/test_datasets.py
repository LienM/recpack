import os
from pathlib import Path
import pytest
from tempfile import NamedTemporaryFile


from recpack import datasets
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


@pytest.fixture()
def demo_data():
    s = """2 1 2\n3 2 1 3\n1 2"""
    return s


def test_fetch_dataset(demo_data):
    """Test that fetch gets the data correctly. Downloading a file when not present,
    not downloading when present, and overwriting when force is True.
    """

    with NamedTemporaryFile() as f:

        p = Path(f.name)
        filename = p.name
        path = str(p.parent)

        # We'll test using citeulike style data
        d = datasets.CiteULike(path=path, filename=filename)

        def download_mock():
            with open(f.name, "w") as fw:
                fw.write(demo_data)

        d._download_dataset = download_mock

        # Fetch dataset will download the file
        # We have to force here, because of the temporary file used
        d.fetch_dataset(force=True)

        # Now file should exist
        assert os.path.exists(f.name)

        # This should load the full DataFrame.
        df = d._load_dataframe()
        assert df.shape == (6, 2)

        # We'll overwrite the file with some other content in the same format
        with open(f.name, "w") as fw:
            fw.write("2 1 2\n")
            fw.write("1 2")

        # Dataframe gets reloaded from file
        df2 = d._load_dataframe()
        assert df2.shape == (3, 2)

        # Fetching with the file already existing does not overwrite the file
        d.fetch_dataset()

        # No changes in DataFrame, since file was not changed
        df2_bis = d._load_dataframe()
        assert df2_bis.shape == df2.shape

        # With the force=True parameter the dataset will be downloaded,
        # overwriting of the already existing file
        d.fetch_dataset(force=True)

        # Dataframe should be the same as the first download
        df_bis = d._load_dataframe()
        assert df_bis.shape == df.shape


def test_ensure_path_exists():
    d = datasets.CiteULike()

    p = Path(d.path)
    assert p.exists()
    p.rmdir()


def test_error_no_default_filename():
    with pytest.raises(ValueError):
        _ = datasets.Dataset()


def test_add_filter(dataset_path):
    filename = "citeulike_sample.dat"

    d = datasets.CiteULike(path=dataset_path, filename=filename)

    d.add_filter(NMostPopular(3, d.ITEM_IX))

    data = d.load()

    assert data.shape[1] == 3


def test_add_filter_w_index(dataset_path):
    filename = "citeulike_sample.dat"

    d = datasets.CiteULike(path=dataset_path, filename=filename)

    d.add_filter(NMostPopular(3, d.ITEM_IX), index=0)

    assert type(d.preprocessor.filters[0]) == NMostPopular
    assert len(d.preprocessor.filters) == 3


def test_citeulike(dataset_path):
    # To get sample we used head -1000 users.dat
    filename = "citeulike_sample.dat"

    d = datasets.CiteULike(path=dataset_path, filename=filename)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX]).all()

    assert df.shape == (36872, 2)
    assert df[d.USER_IX].nunique() == 1000
    assert df[d.ITEM_IX].nunique() == 13689

    data = d.load()

    assert data.shape == (963, 1748)


def test_movielens25m(dataset_path):
    # To get sample we used head -10000 ratings.csv
    filename = "ml-25m_sample.csv"

    d = datasets.MovieLens25M(path=dataset_path, filename=filename)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (9999, 4)
    assert df[d.USER_IX].nunique() == 75
    assert df[d.ITEM_IX].nunique() == 3287

    data = d.load()

    assert data.shape == (75, 260)


def test_movielens25m_no_rating_filters(dataset_path):
    # To get sample we used head -10000 ratings.csv
    filename = "ml-25m_sample.csv"

    d = datasets.MovieLens25M(path=dataset_path, filename=filename, use_default_filters=False)
    d.add_filter(MinRating(1, d.RATING_IX))
    d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
    d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (9999, 4)
    assert df[d.USER_IX].nunique() == 75
    assert df[d.ITEM_IX].nunique() == 3287

    data = d.load()

    assert data.shape == (75, 565)


def test_recsys_challenge_2015(dataset_path):
    # To get sample we used head -1000 ratings.csv
    d = datasets.RecsysChallenge2015(path=dataset_path)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.TIMESTAMP_IX, d.ITEM_IX]).all()
    assert df.shape == (1000, 3)
    assert df[d.USER_IX].nunique() == 272
    assert df[d.ITEM_IX].nunique() == 570

    data = d.load()

    assert data.shape == (83, 26)

    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()


@pytest.mark.parametrize(
    "extra_cols, event_types, num_events, final_shape",
    [
        ([], None, 47, (28, 42)),
        (["category_id", "category_code"], None, 47, (28, 42)),
        #
        ([], ["view"], 47, (28, 42)),
        ([], ["cart"], 26, (7, 21)),
        ([], ["purchase"], 1, (1, 1)),
        ([], ["remove_from_cart"], 26, (4, 16)),
        #
        ([], ["view", "cart"], 47 + 26, (30, 60)),
        ([], ["view", "remove_from_cart"], 47 + 26, (30, 54)),
        ([], ["view", "purchase"], 47 + 1, (28, 43)),
        ([], ["cart", "purchase"], 26 + 1, (7, 21)),
        ([], ["cart", "remove_from_cart"], 26 + 26, (10, 37)),
        ([], ["remove_from_cart", "purchase"], 26 + 1, (4, 17)),
        #
        ([], ["view", "cart", "remove_from_cart"], 47 + 26 + 26, (32, 72)),
        ([], ["view", "cart", "purchase"], 47 + 26 + 1, (30, 60)),
        ([], ["view", "remove_from_cart", "purchase"], 47 + 26 + 1, (30, 55)),
        ([], ["cart", "remove_from_cart", "purchase"], 26 + 26 + 1, (10, 37)),
        #
        ([], ["view", "cart", "remove_from_cart", "purchase"], 100, (32, 72)),
    ],
)
def test_cosmeticsshop(
    dataset_path,
    extra_cols,
    event_types,
    num_events,
    final_shape,
):
    # To get sample we used head -100 2019-Dec.csv
    if event_types is None:
        d = datasets.CosmeticsShop(
            path=dataset_path,
            filename="cosmeticsshop-sample.csv",
            use_default_filters=False,
            extra_cols=extra_cols,
        )
    else:
        d = datasets.CosmeticsShop(
            path=dataset_path,
            filename="cosmeticsshop-sample.csv",
            use_default_filters=False,
            extra_cols=extra_cols,
            event_types=event_types,
        )

    df = d._load_dataframe()
    assert (df.columns == d._columns).all()
    assert df.shape == (num_events, len(d._columns))

    # assert
    data = d.load()

    assert data.shape == final_shape

    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()


def test_cosmeticsshop_bad_event_type(dataset_path):
    with pytest.raises(ValueError):
        _ = datasets.CosmeticsShop(
            path=dataset_path,
            filename="cosmeticsshop-sample.csv",
            use_default_filters=False,
            event_types=["hello"],
        )


@pytest.mark.parametrize(
    "event_types, num_events, final_shape",
    [
        (None, 96, (90, 95)),  # Default = view
        #
        (["view"], 96, (90, 95)),
        (["addtocart"], 3, (3, 3)),
        (["transaction"], 1, (1, 1)),
        #
        (["view", "addtocart"], 96 + 3, (93, 97)),
        (["view", "transaction"], 96 + 1, (91, 96)),
        (["addtocart", "transaction"], 3 + 1, (3, 3)),
        #
        (["view", "addtocart", "transaction"], 96 + 3 + 1, (93, 97)),
    ],
)
def test_retail_rocket(
    dataset_path,
    event_types,
    num_events,
    final_shape,
):
    # To get sample we used head -100 2019-Dec.csv
    if event_types is None:
        d = datasets.RetailRocket(
            path=dataset_path,
            filename="retailrocket-sample.csv",
            use_default_filters=False,
        )
    else:
        d = datasets.RetailRocket(
            path=dataset_path,
            filename="retailrocket-sample.csv",
            use_default_filters=False,
            event_types=event_types,
        )

    df = d._load_dataframe()
    assert (df.columns == d._columns).all()
    assert df.shape == (num_events, len(d._columns))

    # assert
    data = d.load()

    assert data.shape == final_shape

    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()


def test_retail_rocket_bad_event_type(dataset_path):
    with pytest.raises(ValueError):
        _ = datasets.CosmeticsShop(
            path=dataset_path,
            filename="retailrocket-sample.csv",
            use_default_filters=False,
            event_types=["hello"],
        )


def test_dummy_dataset():
    d = datasets.DummyDataset(seed=42)

    # This should do nothing, also not raise an error
    d._download_dataset()

    df = d._load_dataframe()
    df2 = d._load_dataframe()

    # Loading twice gives the same reproducible results
    assert (df[d.USER_IX] == df2[d.USER_IX]).all()
    assert (df[d.ITEM_IX] == df2[d.ITEM_IX]).all()
    assert (df[d.TIMESTAMP_IX] == df2[d.TIMESTAMP_IX]).all()

    assert df.shape[0] == d.num_interactions
    items_kept = (
        (df.drop_duplicates([d.USER_IX, d.ITEM_IX]).groupby(d.ITEM_IX)[d.USER_IX].count() >= 2)
        .reset_index()
        .rename(columns={d.USER_IX: "enough_interactions"})
    )
    df_after_filter_1 = df.merge(items_kept[items_kept.enough_interactions], on=d.ITEM_IX)

    users_removed = (
        df_after_filter_1.drop_duplicates([d.USER_IX, d.ITEM_IX]).groupby(d.USER_IX)[d.ITEM_IX].count() < 2
    ).sum()

    im = d.load()
    assert im.shape == (
        df[d.USER_IX].nunique() - users_removed,
        items_kept.enough_interactions.sum(),
    )
