import os
from pathlib import Path
import pytest
from tempfile import NamedTemporaryFile


from recpack.data import datasets
from recpack.preprocessing.filters import NMostPopular


@pytest.fixture()
def demo_data():
    s = """2 1 2\n3 2 1 3\n1 2"""
    return s


def test_fetch_dataset(demo_data):
    """Test that

    :param demo_data: [description]
    :type demo_data: [type]
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
        df = d.load_dataframe()
        assert df.shape == (6, 2)

        # We'll overwrite the file with some other content in the same format
        with open(f.name, "w") as fw:
            fw.write("2 1 2\n")
            fw.write("1 2")

        # Dataframe gets reloaded from file
        df2 = d.load_dataframe()
        assert df2.shape == (3, 2)

        # Fetching with the file already existing does not overwrite the file
        d.fetch_dataset()

        # No changes in DataFrame, since file was not changed
        df2_bis = d.load_dataframe()
        assert df2_bis.shape == df2.shape

        # With the force=True parameter the dataset will be downloaded,
        # overwriting of the already existing file
        d.fetch_dataset(force=True)

        # Dataframe should be the same as the first download
        df_bis = d.load_dataframe()
        assert df_bis.shape == df.shape


def test_ensure_path_exists():
    d = datasets.CiteULike()

    p = Path(d.path)
    assert p.exists()
    p.rmdir()


def test_error_no_default_filename():
    with pytest.raises(ValueError):
        _ = datasets.Dataset()


def test_add_filter():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")
    filename = "citeulike_sample.dat"

    d = datasets.CiteULike(path=path, filename=filename)

    d.add_filter(NMostPopular(3, d.ITEM_IX))

    data = d.load_interaction_matrix()

    assert data.shape[1] == 3


def test_add_filter_w_index():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")
    filename = "citeulike_sample.dat"

    d = datasets.CiteULike(path=path, filename=filename)

    d.add_filter(NMostPopular(3, d.ITEM_IX), index=0)

    assert type(d.preprocessor.filters[0]) == NMostPopular
    assert len(d.preprocessor.filters) == 3


def test_citeulike():
    # To get sample we used head -1000 users.dat
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")
    filename = "citeulike_sample.dat"

    d = datasets.CiteULike(path=path, filename=filename)

    df = d.load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX]).all()

    assert df.shape == (36872, 2)
    assert df[d.USER_IX].nunique() == 1000
    assert df[d.ITEM_IX].nunique() == 13689

    data = d.load_interaction_matrix()

    assert data.shape == (963, 1748)


def test_movielens25m():
    # To get sample we used head -10000 ratings.csv
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")
    filename = "ml-25m_sample.csv"

    d = datasets.MovieLens25M(path=path, filename=filename)
    print(d.file_path)

    df = d.load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (9999, 4)
    assert df[d.USER_IX].nunique() == 75
    assert df[d.ITEM_IX].nunique() == 3287

    data = d.load_interaction_matrix()

    assert data.shape == (75, 565)


def test_recsys_challenge_2015():
    # To get sample we used head -1000 ratings.csv
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")
    d = datasets.RecsysChallenge2015(path=path)

    df = d.load_dataframe()
    assert (df.columns == [d.USER_IX, d.TIMESTAMP_IX, d.ITEM_IX]).all()
    assert df.shape == (1000, 3)
    assert df[d.USER_IX].nunique() == 272
    assert df[d.ITEM_IX].nunique() == 570

    data = d.load_interaction_matrix()

    assert data.shape == (54, 291)

    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()
