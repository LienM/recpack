# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

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


def test_ensure_path_exists(dataset_path):
    d = datasets.DummyDataset(path=os.path.join(dataset_path, "tmp"))

    p = Path(d.path)
    assert p.exists()
    p.rmdir()


def test_error_no_default_filename():
    with pytest.raises(ValueError):
        _ = datasets.Dataset()


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
