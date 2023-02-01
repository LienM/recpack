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

from recpack.datasets import AdressaOneWeek
from recpack.preprocessing.filters import MinUsersPerItem

MOCK_TAR_FILE_NAME = "one_week.tar.gz"


def mock_download(dataset_path, url, path):
    shutil.copyfile(os.path.join(dataset_path, "compressed", MOCK_TAR_FILE_NAME), path)


@pytest.fixture()
def adressa_dataset(dataset_path):
    """
    zipfile contents were constructed with specific checks in mind:
    20170101 -> contains 3 non view events + 5 view events with a 3rd user
    20170102 -> Contains 5 views with no user_ids -> events should be skipped
    20170103 -> Contains 5 views without a timestamp -> should be skipped
    20170104 -> Contains 5 views with new user
    20170105 -> Contains 5 views with new item
    20170106 -> Contains 5 views with no item -> should be skipped
    20170107 -> Contains 5 base views

    The resulting interaction matrix should contain 3 users with 1 item.
    Each user should have 5 visits of the same item.
    """
    filename = "adressa_views.csv"
    with patch("recpack.datasets.adressa._fetch_remote", partial(mock_download, dataset_path)):
        ds = AdressaOneWeek(dataset_path, filename, use_default_filters=False)
        ds.add_filter(MinUsersPerItem(2, ds.ITEM_IX, ds.USER_IX))

        ds.fetch_dataset(force=True)

        yield ds

    # Clean up the created file
    os.remove(ds.file_path)


def test__load_dataframe(adressa_dataset):
    """Check if loading adressa works properly"""

    df = adressa_dataset._load_dataframe()
    assert (df.columns == [adressa_dataset.USER_IX, adressa_dataset.ITEM_IX, adressa_dataset.TIMESTAMP_IX]).all()

    assert df.shape == (20, 3)
    assert df[adressa_dataset.USER_IX].nunique() == 3
    assert df[adressa_dataset.ITEM_IX].nunique() == 2


def test_load_interaction_matrix(adressa_dataset):

    im = adressa_dataset.load()

    assert im.shape == (3, 1)
    assert im.num_interactions == 15
