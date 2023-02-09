# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack import datasets
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


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
