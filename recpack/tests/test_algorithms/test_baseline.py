import numpy
import pytest
import random
import scipy.sparse
import recpack.algorithms


@pytest.fixture(scope="function")
def data():
    # generate scipy.sparse matrix with user interactions.
    users = list(range(10))
    u_, i_ = [], []
    for user in users:
        items = list(range(10))
        items_interacted = numpy.random.choice(items, 3, replace=False)

        u_.extend([user] * 3)
        i_.extend(items_interacted)
    return scipy.sparse.csr_matrix((numpy.ones(len(u_)), (u_, i_)))


@pytest.fixture(scope="function")
def data_in_out():
    d = []
    for i in range(1, 11):
        users = list(range(i))
        u_in, i_in = [], []
        u_out, i_out = [], []
        for user in users:
            items = list(range(10))
            items_interacted = numpy.random.choice(items, 6, replace=False)
            items_in = items_interacted[:3]
            items_out = items_interacted[3:]

            u_in.extend([user] * 3)
            u_out.extend(([user] * 3))
            i_in.extend(items_in)
            i_out.extend(items_out)

        in_ = scipy.sparse.csr_matrix(
            (numpy.ones(len(u_in)), (u_in, i_in)), shape=(i, 10)
        )
        out_ = scipy.sparse.csr_matrix(
            (numpy.ones(len(u_out)), (u_out, i_out)), shape=(i, 10)
        )

        d.append((in_, out_))
    return d


def test_random(data, data_in_out):

    seed = 42
    K = 5
    algo = recpack.algorithms.Random(K=K, seed=42)
    algo.fit(data)

    for out_, in_ in data_in_out:
        result = algo.predict(in_)
        assert len(result.nonzero()[1]) == result.shape[0] * K
        # TODO: What else to test?


def test_random_use_only_interacted_items(purchases):
    algo = recpack.algorithms.Random(K=2, use_only_interacted_items=True)

    algo.fit(purchases)
    assert (
        len(algo.items_) == 2
    )  # 2 purchased items in the purchases interaction matrix

    algo = recpack.algorithms.Random(K=2, use_only_interacted_items=False)

    algo.fit(purchases)
    assert len(algo.items_) == 5


def test_popularity():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = scipy.sparse.csr_matrix((values, (user_i, item_i)))
    algo = recpack.algorithms.Popularity(K=2)

    algo.fit(train_data)

    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 1], [1, 1])), shape=(5, 5))
    prediction = algo.predict(_in)

    assert (prediction[0] != prediction[1]).nnz == 0
    assert prediction[0, 4] != 0
    assert prediction[0, 3] != 0
    assert prediction[0, 4] > prediction[0, 3]
    assert (prediction[0, :3].toarray() == numpy.array([0, 0, 0])).all()
