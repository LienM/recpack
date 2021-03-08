import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms.samplers import bootstrap_sample_pairs, warp_sample_pairs
from recpack.data.matrix import to_binary


def test_warp_sampling_exact(pageviews):
    pageviews = to_binary(pageviews)
    batch_size = 4
    U = 10
    total_interactions = 0
    for users, pos_interactions, neg_interactions in warp_sample_pairs(
        pageviews, U=U, batch_size=batch_size, exact=True
    ):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)
        assert users.shape[0] == pos_interactions.shape[0]
        assert users.shape[0] == neg_interactions.shape[0]
        assert neg_interactions.shape[1] == U
        total_interactions += users.shape[0]
        # No negative interactions should exist in the original pageviews
        for i in range(U):
            items = neg_interactions.numpy()[:, i].copy()
            interactions = pageviews[users.numpy().copy(), items]
            np.testing.assert_array_almost_equal(interactions, 0)
    assert total_interactions == pageviews.nnz


def test_warp_sampling(pageviews):

    batch_size = 4
    U = 10

    total_interactions = 0

    for users, pos_interactions, neg_interactions in warp_sample_pairs(
        pageviews, U=U, batch_size=batch_size
    ):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)
        assert users.shape[0] == pos_interactions.shape[0]
        assert users.shape[0] == neg_interactions.shape[0]
        assert neg_interactions.shape[1] == U
        total_interactions += users.shape[0]

    assert total_interactions == pageviews.nnz


def test_bootstrap_sampling_exact(pageviews):

    # users = [np.random.randint(0, 2000) for i in range(10000)]
    # items = [np.random.randint(0, 2000) for i in range(10000)]
    # values = [1 for i in range(10000)]
    # pageviews = csr_matrix((values, (users, items)), shape=(2000, 2000))
    # pageviews needs to be binary
    pageviews[pageviews > 1] = 1

    batch_size = 1000
    sample_size = 10000
    total_interactions = 0

    for output in bootstrap_sample_pairs(
        pageviews, batch_size=batch_size, exact=True, sample_size=sample_size
    ):
        np_output = output.numpy()
        b = output.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b
        print(pageviews.toarray())
        print(np_output)
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            pageviews[np_output[:, 0], np_output[:, 2]], 0
        )

    assert total_interactions == sample_size


def test_bootstrap_sampling(pageviews):
    batch_size = 4

    total_interactions = 0

    for output in bootstrap_sample_pairs(pageviews, batch_size=batch_size):
        b = output.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b

    assert total_interactions == pageviews.nnz
