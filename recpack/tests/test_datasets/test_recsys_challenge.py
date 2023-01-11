import pytest

from recpack import datasets
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, NMostPopular, MinRating


def test_recsys_challenge_2015(dataset_path):
    # To get sample we used head -1000 ratings.csv
    d = datasets.RecsysChallenge2015(path=dataset_path)

    df = d._load_dataframe()
    assert (df.columns == [d.USER_IX, d.TIMESTAMP_IX, d.ITEM_IX]).all()
    assert df.shape == (1000, 3)
    assert df[d.USER_IX].nunique() == 272
    assert df[d.ITEM_IX].nunique() == 570

    data = d.load()

    assert data.shape == (83, 26)

    # We can't just download this dataset, since it requires Kaggle access
    with pytest.raises(NotImplementedError):
        d._download_dataset()
