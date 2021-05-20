from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.samplers import (
    BootstrapSampler,
    WarpSampler,
    PositiveNegativeSampler,
)
from recpack.data.matrix import to_binary


def test_warp_sampling_exact():
    users = [np.random.randint(0, 100) for i in range(1000)]
    items = [np.random.randint(0, 25) for i in range(1000)]
    values = [1 for i in range(1000)]
    pageviews = csr_matrix((values, (users, items)), shape=(100, 25))
    pageviews = to_binary(pageviews)

    batch_size = 100
    U = 10

    sampler = WarpSampler(U=U, batch_size=batch_size, exact=True)

    total_interactions = 0
    for users, pos_interactions, neg_interactions in sampler.sample(pageviews):
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

                np.testing.assert_array_equal(overlap, False)

    assert total_interactions == pageviews.nnz


def test_warp_sampling(pageviews):
    pageviews = to_binary(pageviews)
    batch_size = 4
    U = 10

    sampler = WarpSampler(U=U, batch_size=batch_size, exact=False)

    total_interactions = 0
    for users, pos_interactions, neg_interactions in sampler.sample(pageviews):

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

    sampler = BootstrapSampler(batch_size=batch_size, exact=True)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(
        pageviews, sample_size=sample_size
    ):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            pageviews[users.numpy().copy(), negatives_batch.squeeze().numpy().copy()], 0
        )

    assert total_interactions == sample_size


def test_bootstrap_sampling(pageviews):
    batch_size = 4

    sampler = BootstrapSampler(batch_size=batch_size)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(pageviews):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b

    assert total_interactions == pageviews.nnz


def test_sample_positives_and_negatives_bootstrap(pageviews):
    batch_size = 4

    sampler = PositiveNegativeSampler(U=1, batch_size=batch_size, replace=True)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(pageviews):

        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b

    assert total_interactions == pageviews.nnz


def test_sample_positives_and_negatives_bootstrap_exact(pageviews):

    # pageviews needs to be binary
    pageviews = to_binary(pageviews)

    batch_size = 1000
    sample_size = 10000
    sampler = PositiveNegativeSampler(
        U=1, batch_size=batch_size, replace=True, exact=True
    )

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(
        pageviews, sample_size=sample_size
    ):

        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            pageviews[users.numpy().copy(), negatives_batch.squeeze().numpy().copy()], 0
        )

    assert total_interactions == sample_size


def test_sample_positives_and_negatives_warp(pageviews):
    pageviews = to_binary(pageviews)
    batch_size = 4
    U = 10

    sampler = PositiveNegativeSampler(U=U, batch_size=batch_size, replace=False)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(pageviews):

        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)
        assert users.shape[0] == positives_batch.shape[0]
        assert users.shape[0] == negatives_batch.shape[0]
        assert negatives_batch.shape[1] == U
        total_interactions += users.shape[0]

    assert total_interactions == pageviews.nnz


def test_sample_positives_and_negatives_w_positives_arg(larger_matrix):

    # pageviews needs to be binary
    pageviews = to_binary(larger_matrix)

    all_positives = np.array(pageviews.nonzero()).T
    # Select first 100 entries as positives to sample from
    selected_positives = all_positives[0:100, :]

    selected_positives_aslist = selected_positives.tolist()

    batch_size = 12

    sampler = PositiveNegativeSampler(
        U=1, batch_size=batch_size, replace=False, exact=True
    )

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(
        pageviews, positives=selected_positives
    ):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            pageviews[users.numpy().copy(), negatives_batch.squeeze().numpy().copy()], 0
        )

        for i in range(0, positives_batch.shape[0]):
            user = users[i]
            positive = positives_batch[i]

            assert [user, positive] in selected_positives_aslist

    assert total_interactions == selected_positives.shape[0]


def test_sample_positives_and_negatives_w_unigram(mat):
    # needs to be binary
    pageviews = mat.binary_values

    batch_size = 1000
    sample_size = 10000
    sampler = PositiveNegativeSampler(
        U=1, batch_size=batch_size, replace=True, exact=False, distribution="unigram"
    )

    negatives_counts = np.zeros(5)
    for users, positives_batch, negatives_batch in sampler.sample(
        pageviews, sample_size=sample_size
    ):
        for n in negatives_batch.numpy():
            negatives_counts[n[0]] += 1

    negatives_perc = negatives_counts / negatives_counts.sum()
    # Items visited in pageviews: [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2]
    assert negatives_perc[0] > negatives_perc[3]
    assert negatives_perc[0] > negatives_perc[4]

    np.testing.assert_almost_equal(negatives_perc[0], negatives_perc[1], decimal=2)
    np.testing.assert_almost_equal(negatives_perc[0], negatives_perc[2], decimal=2)
    np.testing.assert_almost_equal(negatives_perc[3], negatives_perc[4], decimal=2)
