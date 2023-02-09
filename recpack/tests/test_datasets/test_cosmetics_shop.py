# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

from recpack import datasets
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


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
            filename="cosmeticsshop-sample.zip",
            use_default_filters=False,
            extra_cols=extra_cols,
        )
    else:
        d = datasets.CosmeticsShop(
            path=dataset_path,
            filename="cosmeticsshop-sample.zip",
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
            filename="cosmeticsshop-sample.zip",
            use_default_filters=False,
            event_types=["hello"],
        )
