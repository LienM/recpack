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
    "event_types, num_events, final_shape",
    [
        (None, 96, (90, 95)),  # Default = view
        #
        (["view"], 96, (90, 95)),
        (["addtocart"], 3, (3, 3)),
        (["transaction"], 1, (1, 1)),
        #
        (["view", "addtocart"], 96 + 3, (93, 97)),
        (["view", "transaction"], 96 + 1, (91, 96)),
        (["addtocart", "transaction"], 3 + 1, (3, 3)),
        #
        (["view", "addtocart", "transaction"], 96 + 3 + 1, (93, 97)),
    ],
)
def test_retail_rocket(
    dataset_path,
    event_types,
    num_events,
    final_shape,
):
    # To get sample we used head -100 2019-Dec.csv
    if event_types is None:
        d = datasets.RetailRocket(
            path=dataset_path,
            filename="retailrocket-sample.csv",
            use_default_filters=False,
        )
    else:
        d = datasets.RetailRocket(
            path=dataset_path,
            filename="retailrocket-sample.csv",
            use_default_filters=False,
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


def test_retail_rocket_bad_event_type(dataset_path):
    with pytest.raises(ValueError):
        _ = datasets.CosmeticsShop(
            path=dataset_path,
            filename="retailrocket-sample.csv",
            use_default_filters=False,
            event_types=["hello"],
        )