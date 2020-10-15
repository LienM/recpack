import pytest

from recpack.algorithms.metric_learning.CML import warp_sample_pairs


def test_warp_sampling(pageviews):

    batch_size = 4
    U = 10

    total_interactions = 0

    for users, pos_interactions, neg_interactions in warp_sample_pairs(pageviews, U=U, batch_size=batch_size):
        b = users.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)
        assert users.shape[0] == pos_interactions.shape[0]
        assert users.shape[0] == neg_interactions.shape[0]
        assert neg_interactions.shape[1] == U
        total_interactions += users.shape[0]

    assert total_interactions == pageviews.nnz
