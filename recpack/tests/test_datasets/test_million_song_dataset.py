# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from pandas.testing import assert_frame_equal
import pytest


from recpack import datasets
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


@pytest.mark.parametrize(
    "user_id, item_id, count",
    [
        ("b80344d063b5ccb3212f76538f3d9e43d87dca9e", "SODDNQT12A6D4F5F7E", 5),
        ("b80344d063b5ccb3212f76538f3d9e43d87dca9e", "SOBSUJE12A6D4F8CF5", 2),
        ("b80344d063b5ccb3212f76538f3d9e43d87dca9e", "SOBYHAJ12A6701BF1D", 1),
        ("b80344d063b5ccb3212f76538f3d9e43d87dca9e", "SOMZWUW12A8C1400BC", 6),
    ],
)
def test_msd_unwrap_count(user_id, item_id, count, dataset_path):
    # To get sample we used head -10000 train_triplets.tsv
    filename = "msd_train_triplets_sample.csv"

    d = datasets.MillionSongDataset(path=dataset_path, filename=filename)

    df = d._load_dataframe()

    rows = df[(df[d.USER_IX] == user_id) & (df[d.ITEM_IX] == item_id)]

    assert rows.shape[0] == count


def test_msd_assert_columns(dataset_path):
    # Test whether more stringent filters leads to a smaller dataset
    # To get sample we used head -10000 train_triplets.tsv
    filename = "msd_train_triplets_sample.csv"

    d = datasets.MillionSongDataset(path=dataset_path, filename=filename, use_default_filters=False)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX]).all()


def test_msd_filters(dataset_path):
    # Test whether more stringent filters leads to a smaller dataset
    # To get sample we used head -10000 train_triplets.tsv
    filename = "msd_train_triplets_sample.csv"

    d1 = datasets.MillionSongDataset(path=dataset_path, filename=filename, use_default_filters=False)
    d1.add_filter(MinItemsPerUser(50, d1.ITEM_IX, d1.USER_IX))

    d2 = datasets.MillionSongDataset(path=dataset_path, filename=filename, use_default_filters=False)
    d2.add_filter(MinItemsPerUser(20, d2.ITEM_IX, d2.USER_IX))

    df1 = d1._load_dataframe()
    df2 = d2._load_dataframe()

    assert_frame_equal(df1, df2)

    data1 = d1.load()
    data2 = d2.load()

    assert data1.shape <= data2.shape
    assert data1.values.nnz < data2.values.nnz


def test_tasteprofile_assert_columns(dataset_path):
    # Test whether more stringent filters leads to a smaller dataset
    # To get sample we used head -10000 train_triplets.tsv
    filename = "msd_train_triplets_sample.csv"

    d = datasets.TasteProfile(path=dataset_path, filename=filename, use_default_filters=False)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.ITEM_IX]).all()
