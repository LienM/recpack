# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from functools import partial
import os
import pytest
import shutil
from unittest.mock import patch

from recpack.datasets import Netflix

MOCK_TAR_FILE_NAME = "nf_prize_dataset.tar.gz"


def mock_download(dataset_path, url, path):
    shutil.copyfile(os.path.join(dataset_path, "compressed", MOCK_TAR_FILE_NAME), path)


@pytest.fixture()
def netflix_dataset(dataset_path):
    with patch("recpack.datasets.netflix._fetch_remote", partial(mock_download, dataset_path)):
        res = Netflix(path=dataset_path)
        res.fetch_dataset(force=True)
        yield res

    # clean up the csv file
    if os.path.exists(res.file_path):
        os.remove(res.file_path)


def test__load_dataframe(netflix_dataset):
    df = netflix_dataset._load_dataframe()
    assert (df.columns == [netflix_dataset.USER_IX, netflix_dataset.ITEM_IX, netflix_dataset.TIMESTAMP_IX, netflix_dataset.RATING_IX]).all()

    # We added 21 events total to the dummy files & have 4 columns
    assert df.shape == (21, 4)
    # 6 unique users
    assert df[netflix_dataset.USER_IX].nunique() == 6
    # 5 unique items
    assert df[netflix_dataset.ITEM_IX].nunique() == 5


def test_load_interaction_matrix(netflix_dataset):

    im = netflix_dataset.load()

    # all items have at least 1 interaction
    # only user 1 and 2 make it through the selection
    # User 3 has 5 events, but not enough with rating 4 or higher.
    assert im.shape == (2, 5)
    assert im.num_interactions == 10
