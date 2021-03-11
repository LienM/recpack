import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms.samplers import bootstrap_sample_pairs, warp_sample_pairs
from recpack.data.matrix import to_binary


def test_warp_sampling_exact():
    users = [np.random.randint(0, 100) for i in range(1000)]
    items = [np.random.randint(0, 25) for i in range(1000)]
    values = [1 for i in range(1000)]
    pageviews = csr_matrix((values, (users, items)), shape=(100, 25))
    pageviews = to_binary(pageviews)

    batch_size = 100
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

        # There should be no collisions between columns of negative samples
        for i in range(U):
            for j in range(i):

                overlap = (
                    neg_interactions[:, j].numpy() == neg_interactions[:, i].numpy()
                )
                print(
                    np.stack(
                        (
                            neg_interactions[:, j].numpy(),
                            neg_interactions[:, i].numpy(),
                            overlap,
                        ),
                        axis=1,
                    )
                )

                np.testing.assert_array_equal(overlap, False)

    assert total_interactions == pageviews.nnz


def test_warp_sampling(pageviews):
    pageviews = to_binary(pageviews)
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

    # pageviews needs to be binary
    pageviews = to_binary(pageviews)

    batch_size = 1000
    sample_size = 10000
    total_interactions = 0

    for users, positives_batch, negatives_batch in bootstrap_sample_pairs(
        pageviews, batch_size=batch_size, exact=True, sample_size=sample_size
    ):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            pageviews[users.numpy().copy(), negatives_batch.numpy().copy()], 0
        )

    assert total_interactions == sample_size


def test_bootstrap_sampling(pageviews):
    batch_size = 4

    total_interactions = 0

    for users, positives_batch, negatives_batch in bootstrap_sample_pairs(
        pageviews, batch_size=batch_size
    ):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b

    assert total_interactions == pageviews.nnz
