import recpack.algorithms
import scipy.sparse


def test_item_knn():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    data = scipy.sparse.csr_matrix((values, (users, items)))

    algo = recpack.algorithms.get_algorithm("itemKNN")(K=2)

    algo.fit(data)
    assert algo.item_cosine_similarities.toarray() is False

    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 1], [1, 1])), shape=(3, 3))
    result = algo.predict(_in)
    assert result is False

