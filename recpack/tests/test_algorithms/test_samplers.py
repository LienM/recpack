# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms.samplers import (
    BootstrapSampler,
    WarpSampler,
    PositiveNegativeSampler,
    unigram_distribution,
    SequenceMiniBatchPositivesTargetsNegativesSampler,
)
from recpack.matrix import to_binary


@pytest.mark.parametrize("num_negatives, batch_size", [(1, 3), (3, 1), (3, 2), (1, 1), (6, 6), (100, 6), (0, 6)])
def test_sequence_mini_batch_pos_tar_neg_sampling(matrix_sessions, num_negatives, batch_size):
    pad_token = matrix_sessions.shape[1] + 1

    sampler = SequenceMiniBatchPositivesTargetsNegativesSampler(num_negatives, pad_token, batch_size=batch_size)

    total_interactions = 0
    total_users = 0
    for uid_batch, pos_batch, tar_batch, neg_batch in sampler.sample(matrix_sessions):
        # Check batch_size
        b = uid_batch.shape[0]
        assert (b == batch_size) or (b == matrix_sessions.num_interactions % batch_size)
        assert pos_batch.shape[0] == uid_batch.shape[0]
        assert neg_batch.shape[0] == uid_batch.shape[0]
        assert pos_batch.shape == tar_batch.shape

        # Check sequence length
        assert pos_batch.shape[1] == neg_batch.shape[1]

        # Check number of negatives
        assert neg_batch.shape[2] == num_negatives

        total_users += uid_batch.shape[0]
        # Sum all interactions that are not pads
        total_interactions += (pos_batch != pad_token).sum()

        # All but last item should match rolled positives
        np.testing.assert_array_almost_equal(np.roll(pos_batch, -1, axis=1)[:, :-1], tar_batch[:, :-1])
        # Last item should be padding
        assert (tar_batch[:, -1] == pad_token).all()

        # Negative samples should never match the target sample in the same location in the sequence.
        for i in range(num_negatives):
            negative_sequences = neg_batch.detach().numpy()[:, :, i]
            tar_sequences = tar_batch.detach().numpy()

            assert not (negative_sequences == tar_sequences).any()

    # Assert all interactions & users were present
    assert total_interactions == matrix_sessions.num_interactions
    assert total_users == matrix_sessions.num_active_users


def test_warp_sampling_exact():
    users = [np.random.randint(0, 100) for i in range(500)]
    items = [np.random.randint(0, 25) for i in range(500)]
    values = [1 for i in range(500)]
    X_in = csr_matrix((values, (users, items)), shape=(100, 25))
    X_in = to_binary(X_in)

    batch_size = 100
    num_negatives = 5

    sampler = WarpSampler(num_negatives=num_negatives, batch_size=batch_size, exact=True)

    total_interactions = 0
    for users, pos_interactions, neg_interactions in sampler.sample(X_in):
        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)
        assert users.shape[0] == pos_interactions.shape[0]
        assert users.shape[0] == neg_interactions.shape[0]
        assert neg_interactions.shape[1] == num_negatives
        total_interactions += users.shape[0]
        # No negative interactions should exist in the original X_in
        for i in range(num_negatives):
            items = neg_interactions.numpy()[:, i].copy()
            interactions = X_in[users.numpy().copy(), items]
            np.testing.assert_array_almost_equal(interactions, 0)

        # There should be no collisions between columns of negative samples
        for i in range(num_negatives):
            for j in range(i):

                overlap = neg_interactions[:, j].numpy() == neg_interactions[:, i].numpy()

                np.testing.assert_array_equal(overlap, False)

    assert total_interactions == X_in.nnz


def test_warp_sampling(X_in):
    X_in = to_binary(X_in)
    batch_size = 4
    num_negatives = 10

    sampler = WarpSampler(num_negatives=num_negatives, batch_size=batch_size, exact=False)

    total_interactions = 0
    for users, pos_interactions, neg_interactions in sampler.sample(X_in):

        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)
        assert users.shape[0] == pos_interactions.shape[0]
        assert users.shape[0] == neg_interactions.shape[0]
        assert neg_interactions.shape[1] == num_negatives
        total_interactions += users.shape[0]

    assert total_interactions == X_in.nnz


def test_bootstrap_sampling_exact(X_in):

    # X_in needs to be binary
    X_in = to_binary(X_in)

    batch_size = 1000
    sample_size = 10000

    sampler = BootstrapSampler(batch_size=batch_size, exact=True)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(X_in, sample_size=sample_size):
        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            X_in[users.numpy().copy(), negatives_batch.squeeze().numpy().copy()], 0
        )

    assert total_interactions == sample_size


def test_bootstrap_sampling(X_in):
    batch_size = 4

    sampler = BootstrapSampler(batch_size=batch_size)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(X_in):
        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)

        total_interactions += b

    assert total_interactions == X_in.nnz


def test_sample_positives_and_negatives_bootstrap(X_in):
    batch_size = 4

    sampler = PositiveNegativeSampler(num_negatives=1, batch_size=batch_size, replace=True)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(X_in):

        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)

        total_interactions += b

    assert total_interactions == X_in.nnz


def test_sample_positives_and_negatives_bootstrap_exact(X_in):

    # X_in needs to be binary
    X_in = to_binary(X_in)

    batch_size = 1000
    sample_size = 10000
    sampler = PositiveNegativeSampler(num_negatives=1, batch_size=batch_size, replace=True, exact=True)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(X_in, sample_size=sample_size):

        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            X_in[users.numpy().copy(), negatives_batch.squeeze().numpy().copy()], 0
        )

    assert total_interactions == sample_size


def test_sample_positives_and_negatives_warp(X_in):
    X_in = to_binary(X_in)
    batch_size = 4
    num_negatives = 10

    sampler = PositiveNegativeSampler(num_negatives=num_negatives, batch_size=batch_size, replace=False)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(X_in):

        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)
        assert users.shape[0] == positives_batch.shape[0]
        assert users.shape[0] == negatives_batch.shape[0]
        assert negatives_batch.shape[1] == num_negatives
        total_interactions += users.shape[0]

    assert total_interactions == X_in.nnz


def test_sample_positives_and_negatives_w_positives_arg(larger_matrix):

    # X_in needs to be binary
    X_in = to_binary(larger_matrix)

    all_positives = np.array(X_in.nonzero()).T
    # Select first 100 entries as positives to sample from
    selected_positives = all_positives[0:100, :]

    selected_positives_aslist = selected_positives.tolist()

    batch_size = 12

    sampler = PositiveNegativeSampler(num_negatives=1, batch_size=batch_size, replace=False, exact=True)

    total_interactions = 0
    for users, positives_batch, negatives_batch in sampler.sample(X_in, positives=selected_positives):
        b = users.shape[0]
        assert (b == batch_size) or (b == X_in.nnz % batch_size)

        total_interactions += b
        # No negatives should be accidental positives
        np.testing.assert_array_almost_equal(
            X_in[users.numpy().copy(), negatives_batch.squeeze().numpy().copy()], 0
        )

        for i in range(0, positives_batch.shape[0]):
            user = users[i]
            positive = positives_batch[i]

            assert [user, positive] in selected_positives_aslist

    assert total_interactions == selected_positives.shape[0]


def test_sample_positives_and_negatives_w_unigram(mat):
    # needs to be binary
    X_in = mat.binary_values

    batch_size = 1000
    sample_size = 10000
    sampler = PositiveNegativeSampler(
        num_negatives=1, batch_size=batch_size, replace=True, exact=False, distribution="unigram"
    )

    negatives_counts = np.zeros(5)
    for users, positives_batch, negatives_batch in sampler.sample(X_in, sample_size=sample_size):
        for n in negatives_batch.numpy():
            negatives_counts[n[0]] += 1

    negatives_perc = negatives_counts / negatives_counts.sum()
    # Items visited in X_in: [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2]
    assert negatives_perc[0] > negatives_perc[3]
    assert negatives_perc[0] > negatives_perc[4]

    np.testing.assert_almost_equal(negatives_perc[0], negatives_perc[1], decimal=2)
    np.testing.assert_almost_equal(negatives_perc[0], negatives_perc[2], decimal=2)
    np.testing.assert_almost_equal(negatives_perc[3], negatives_perc[4], decimal=2)


def test_unigram_distribution():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [1, 1, 1, 1, 1, 1],
    )

    X_in = csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    distr = unigram_distribution(X_in)

    np.testing.assert_almost_equal(distr.sum(), 1)

    # 2 seen twice, rest once
    denum = (2 ** 0.75) + 4
    np.testing.assert_almost_equal(distr[0], 1 / denum)
    np.testing.assert_almost_equal(distr[1], 1 / denum)
    np.testing.assert_almost_equal(distr[2], 1 / denum)
    np.testing.assert_almost_equal(distr[3], 2 ** 0.75 / denum)
    np.testing.assert_almost_equal(distr[4], 1 / denum)


def test_distribution_check():
    with pytest.raises(ValueError):
        sampler = PositiveNegativeSampler(distribution="nonexistant")
