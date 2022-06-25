"""The dataset module allows users to easily use to publicly available datasets in their experiments.


.. autosummary::
    :toctree: generated/

    datasets.Dataset
    datasets.DummyDataset
    datasets.CiteULike
    datasets.MovieLens25M
    datasets.RecsysChallenge2015
    datasets.ThirtyMusicSessions
    datasets.CosmeticsShop
    datasets.RetailRocket

Example
~~~~~~~~~

Loading a dataset only takes a couple of lines.
If the file specified does not exist, the dataset is downloaded and written into this file.
Subsequent loading of the dataset then happens from this file. ::

    from recpack.datasets import MovieLens25M

    # Folder needs to exist, file will be downloaded if not present
    # This can take a while
    ml_loader = MovieLens25M(path='datasets/' filename='ml-25m.csv')
    data = ml_loader.load_interaction_matrix()

Each dataset has its own default preprocessing steps, documented in the classes respectively.
To use custom preprocessing a couple more lines should be added to the example. ::

    from recpack.datasets import MovieLens25M
    from recpack.preprocessing.filters import MinRating, MinUsersPerItem, MinItemsPerUser

    ml_loader = MovieLens25M('datasets/ml-25m.csv', preprocess_default=False)
    # Only consider ratings 4 or higher as interactions
    ml_loader.add_filter(MinRating(
        4,
        ml_loader.RATING_IX,
    ))
    # Keep users with at least 5 interactions
    ml_loader.add_filter(MinItemsPerUser(
        5,
        ml_loader.ITEM_IX,
        ml_loader.USER_IX,
    ))
    # Keep items with at least 30 interactions
    ml_loader.add_filter(MinUsersPerItem(
        30,
        ml_loader.ITEM_IX,
        ml_loader.USER_IX,
    ))

    data = ml_loader.load_interaction_matrix()

For an overview of available filters see :ref:`recpack.preprocessing`
"""


from recpack.datasets.base import Dataset
from recpack.datasets.adressa import AdressaOneWeek
from recpack.datasets.cite_u_like import CiteULike
from recpack.datasets.cosmetics_shop import CosmeticsShop
from recpack.datasets.dummy_dataset import DummyDataset
from recpack.datasets.movielens import MovieLens25M
from recpack.datasets.recsys_challenge import RecsysChallenge2015
from recpack.datasets.retail_rocket import RetailRocket
from recpack.datasets.thirty_music_sessions import ThirtyMusicSessions
