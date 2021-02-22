recpack.data.matrix
==========================
.. currentmodule:: recpack.data.matrix

Module with classes for representing data.

Summary
---------
.. autosummary::

      InteractionMatrix

Example
---------
An InteractionMatrix object can be constructed from a dataframe with a row for each interaction. 
The ``item`` and ``user`` values will be indices in the resulting matrix.
The following example constructs a 4x4 matrix, with 4 nonzero values::

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
