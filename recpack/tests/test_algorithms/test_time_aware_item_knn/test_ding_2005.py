import numpy as np
from recpack.algorithms import TARSItemKNNDing, ItemKNN


def test_fit_no_weigting(mat):
    tars_algo = TARSItemKNNDing(K=2, predict_decay=0.5, similarity="cosine")
    item_knn = ItemKNN(K=2, similarity="cosine")

    tars_algo.fit(mat)
    item_knn.fit(mat)

    np.testing.assert_array_equal(item_knn.similarity_matrix_.toarray(), tars_algo.similarity_matrix_.toarray())
