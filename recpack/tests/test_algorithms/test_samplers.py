import pytest

from recpack.algorithms.metric_learning.CML import warp_sample_pairs


def test_warp_sampling(pageviews):

    batch_size = 4
    U = 10

    res = warp_sample_pairs(pageviews, U=U, batch_size=batch_size)

    total_interactions = 0

    for batch in res:
        b = batch.shape[0]
        assert (b == batch_size) or (b == pageviews.nnz % batch_size)
        assert batch.shape[1] == U + 2
        total_interactions += batch.shape[0]

    assert total_interactions == pageviews.nnz
