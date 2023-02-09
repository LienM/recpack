# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack import datasets
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


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


def test_movielens100k(dataset_path):
    # To get sample we used head -10000 u.data
    filename = "ml-100k_sample.data"

    d = datasets.MovieLens100K(path=dataset_path, filename=filename)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (10000, 4)
    assert df[d.USER_IX].nunique() == 385
    assert df[d.ITEM_IX].nunique() == 1254

    data = d.load()

    assert data.shape == (373, 390)


def test_movielens100k_no_rating_filters(dataset_path):
    # To get sample we used head -10000 u.data
    filename = "ml-100k_sample.data"

    d = datasets.MovieLens100K(path=dataset_path, filename=filename, use_default_filters=False)
    d.add_filter(MinRating(1, d.RATING_IX))
    d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
    d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (10000, 4)
    assert df[d.USER_IX].nunique() == 385
    assert df[d.ITEM_IX].nunique() == 1254

    data = d.load()

    assert data.shape == (373, 630)


def test_movielens1m(dataset_path):
    # To get sample we used head -10000 ratings.dat
    filename = "ml-1m_sample.dat"

    d = datasets.MovieLens1M(path=dataset_path, filename=filename)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (10000, 4)
    assert df[d.USER_IX].nunique() == 70
    assert df[d.ITEM_IX].nunique() == 2159

    data = d.load()

    assert data.shape == (70, 405)


def test_movielens1m_no_rating_filters(dataset_path):
    # To get sample we used head -10000 ratings.dat
    filename = "ml-1m_sample.dat"

    d = datasets.MovieLens1M(path=dataset_path, filename=filename, use_default_filters=False)
    d.add_filter(MinRating(1, d.RATING_IX))
    d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
    d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (10000, 4)
    assert df[d.USER_IX].nunique() == 70
    assert df[d.ITEM_IX].nunique() == 2159

    data = d.load()

    assert data.shape == (70, 722)


def test_movielens10m(dataset_path):
    # To get sample we used head -10000 ratings.dat
    filename = "ml-10m_sample.dat"

    d = datasets.MovieLens10M(path=dataset_path, filename=filename)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (10000, 4)
    assert df[d.USER_IX].nunique() == 84
    assert df[d.ITEM_IX].nunique() == 2976

    data = d.load()

    assert data.shape == (83, 324)


def test_movielens10m_no_rating_filters(dataset_path):
    # To get sample we used head -10000 ratings.dat
    filename = "ml-10m_sample.dat"

    d = datasets.MovieLens10M(path=dataset_path, filename=filename, use_default_filters=False)
    d.add_filter(MinRating(1, d.RATING_IX))
    d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
    d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX, d.RATING_IX, d.TIMESTAMP_IX]).all()

    assert df.shape == (10000, 4)
    assert df[d.USER_IX].nunique() == 84
    assert df[d.ITEM_IX].nunique() == 2976

    data = d.load()

    assert data.shape == (84, 590)
