recpack.data.matrix
==========================
.. currentmodule:: recpack.data.datasets

Module with classes for representing data.

Summary
---------
.. autosummary::

      InteractionMatrix

Example
---------
Constructing an InteractionMatrix ::

   from recpack.data.matrix import InteractionMatrix
   data = {
      "user": [3, 2, 1, 1],
      "item": [1, 1, 2, 3],
      "timestamp": [1613736000, 1613736300, 1613736600, 1613736900]
   }
   df = pd.DataFrame.from_dict(data)
   demo_dataframe = InteractionMatrix(df, "item", "user", "timestamp")

Classes
---------

.. automodule:: recpack.data.matrix
   :members:
   :undoc-members:
   :show-inheritance:
