recpack.data.datasets
=======================

.. currentmodule:: recpack.data.datasets

Summary
---------

.. autosummary::
   
      Dataset
      CiteULike
      MovieLens25M
      RecsysChallenge2015

Example
---------
Loading a dataset only takes a couple of lines. ::

      from recpack.data.datasets import MovieLens25M

      ml_loader = MovieLens25M('datasets/ml/ratings.csv')
      data = ml_loader.load_interaction_matrix()

Each dataset has its own default preprocessing steps, documented in the classes respectively.
To use custom preprocessing a couple more lines should be added to the example. ::

      from recpack.data.datasets import MovieLens25M
      from recpack.preprocessing.filters import MinRating, MinUsersPerItem, MinItemsPerUser

      ml_loader = MovieLens25M('datasets/ml/ratings.csv', preprocess_default=False)
      ml_loader.add_filter(MinRating(
            "rating",
            ml_loader.ITEM_IX,
            ml_loader.USER_IX,
            min_rating=1
      ))
      ml_loader.add_filter(MinItemsPerUser(
            5,
            ml_loader.ITEM_IX,
            ml_loader.USER_IX,
      ))
      ml_loader.add_filter(MinUsersPerItem(
            30,
            ml_loader.ITEM_IX,
            ml_loader.USER_IX,
      ))

      data = ml_loader.load_interaction_matrix()

Classes
---------

.. automodule:: recpack.data.datasets
   :members:
   :undoc-members:
