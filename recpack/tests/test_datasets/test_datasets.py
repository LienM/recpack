import os
import pytest

from recpack.data import datasets
from recpack.preprocessing.filters import NMostPopular


def test_fetch_dataset():
    file_name = "/tmp/citeulike.dat"
    # We'll test using citeulike
    d = datasets.CiteULike("/tmp/citeulike.dat")

    # The file should not exist yet.
    assert not os.path.exists(file_name)

    # Fetch dataset will download the file
    d.fetch_dataset()

    # Now file should exist
    assert os.path.exists(file_name)

    # This should load the full dataframe.
    df = d.load_dataframe()
    assert df.shape == (204986, 2)

    # We'll overwrite the file with some other content in the same format
    with open(file_name, "w") as f:
        f.write("2 1 2\n")
        f.write("1 2")

    # Dataframe gets reloaded from file
    df2 = d.load_dataframe()
    assert df2.shape == (3, 2)

    # Fetching with the file already existing does not overwrite the file
    d.fetch_dataset()

    # No changes in dataframe, since file was not changed
    df2_bis = d.load_dataframe()
    assert df2_bis.shape == df2.shape

    # With the force=True parameter the dataset will be downloaded,
    # overwriting of the already existing file
    d.fetch_dataset(force=True)

    # Dataframe should be the same as the first download
    df_bis = d.load_dataframe()
    assert df_bis.shape == df.shape

    # Clean up again
    os.remove(file_name)


def test_add_filter():
    path_to_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "datasets/citeulike_sample.dat"
    )

    d = datasets.CiteULike(path_to_file)

    d.add_filter(NMostPopular(3, d.ITEM_IX, d.USER_IX))

    data = d.load_interaction_matrix()

    assert data.shape[1] == 3


def test_citeulike():
    # To get sample we used head -1000 users.dat
    path_to_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "datasets/citeulike_sample.dat"
    )

    d = datasets.CiteULike(path_to_file)

    df = d.load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX]).all()

    assert df.shape == (36872, 2)
    assert df[d.USER_IX].nunique() == 1000
    assert df[d.ITEM_IX].nunique() == 13689

    data = d.load_interaction_matrix()

    assert data.shape == (963, 1748)


def test_movielens25m():
    # To get sample we used head -10000 ratings.csv
    path_to_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "datasets/ml-25m_sample.csv"
    )

    d = datasets.MovieLens25M(path_to_file)

    df = d.load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (9999, 4)
    assert df[d.USER_IX].nunique() == 75
    assert df[d.ITEM_IX].nunique() == 3287

    data = d.load_interaction_matrix()

    assert data.shape == (75, 573)


def test_recsys_challenge_2015():
    # To get sample we used head -1000 ratings.csv
    path_to_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "datasets/yoochoose-clicks_sample.dat",
    )

    d = datasets.RecsysChallenge2015(path_to_file)

    df = d.load_dataframe()
    assert (df.columns == [d.USER_IX, d.TIMESTAMP_IX, d.ITEM_IX]).all()
    assert df.shape == (1000, 3)
    assert df[d.USER_IX].nunique() == 272
    assert df[d.ITEM_IX].nunique() == 570

    data = d.load_interaction_matrix()

    assert data.shape == (15, 20)

    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()
