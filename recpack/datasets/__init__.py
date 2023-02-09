# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""The dataset module allows users to easily use to publicly available datasets in their experiments.

.. currentmodule:: recpack.datasets


.. autosummary::
    :toctree: generated/

    Dataset
    DummyDataset
    AdressaOneWeek
    CiteULike
    Globo
    MovieLens100K
    MovieLens1M
    MovieLens10M
    MovieLens25M
    Netflix
    RecsysChallenge2015
    ThirtyMusicSessions
    CosmeticsShop
    RetailRocket
    MillionSongDataset
    TasteProfile

Example
~~~~~~~~~

Loading a dataset only takes a couple of lines.
If the file specified does not exist, the dataset is downloaded and written into this file.
Subsequent loading of the dataset then happens from this file. ::

    from recpack.datasets import MovieLens25M

    # Folder needs to exist, file will be downloaded if not present
    # This can take a while
    ml_loader = MovieLens25M(path='datasets/' filename='ml-25m.csv')
    data = ml_loader.load()

Each dataset has its own default preprocessing steps, documented in the classes respectively.
To use custom preprocessing a couple more lines should be added to the example. ::

    from recpack.datasets import MovieLens25M
    from recpack.preprocessing.filters import MinRating, MinUsersPerItem, MinItemsPerUser

    ml_loader = MovieLens25M(path='datasets/', filename='ml-25m.csv', use_default_filters=False)
    # Consider ratings 2 or higher as interactions
    ml_loader.add_filter(MinRating(
        2,
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

    data = ml_loader.load()

For an overview of available filters see :mod:`recpack.preprocessing`
"""


from recpack.datasets.base import Dataset
from recpack.datasets.adressa import AdressaOneWeek
from recpack.datasets.cite_u_like import CiteULike
from recpack.datasets.cosmetics_shop import CosmeticsShop
from recpack.datasets.dummy_dataset import DummyDataset
from recpack.datasets.globo import Globo
from recpack.datasets.movielens import MovieLens100K, MovieLens1M, MovieLens10M, MovieLens25M
from recpack.datasets.netflix import Netflix
from recpack.datasets.recsys_challenge import RecsysChallenge2015
from recpack.datasets.retail_rocket import RetailRocket
from recpack.datasets.thirty_music_sessions import ThirtyMusicSessions
from recpack.datasets.million_song_dataset import MillionSongDataset, TasteProfile
