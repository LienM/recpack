# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

from recpack.datasets import Globo
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


@pytest.mark.parametrize(
    "preprocess, num_events, final_shape",
    [
        (False, 100, (11, 11)),
        (True, 70, (11, 6)),
    ],
)
def test_globo(
    dataset_path,
    preprocess,
    num_events,
    final_shape,
):
    # We created a random dataset, with the same columns
    d = Globo(
        path=dataset_path,
        filename="globo-sample.zip",
        use_default_filters=preprocess,
    )

    data = d.load()

    assert data.shape == final_shape
    assert data.num_interactions == num_events

def test_does_not_download():
    d = Globo()
    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()

