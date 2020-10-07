import pytest
import sklearn

from recpack.algorithms.similarity.BPRMF import BPRMF, bootstrap_sample_pairs
from recpack.algorithms import SVD


def test_bprmf(pageviews):
    a = BPRMF(num_components=2, num_epochs=2, sample_size=10, batch_size=1)
    a.fit(pageviews)

    pred = a.predict(pageviews)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(pageviews.nonzero()[0])


def test_pairwise_ranking(pageviews_for_pairwise):
    """Tests that the pairwise ranking of 2 items is correctly computed."""
    # TODO This should be a different test?
    # b = SVD(2)
    # b.fit(pageviews_for_pairwise)
    # pred = b.predict(pageviews_for_pairwise)

    # assert pred[2, 2] > pred[2, 4]

    # TODO Don't run 50 epochs in a test
    a = BPRMF(num_components=2, num_epochs=2, sample_size=10, batch_size=1)

    a.fit(pageviews_for_pairwise)
    # print(pageviews_for_pairwise.toarray())
    pred = a.predict(pageviews_for_pairwise)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(pageviews_for_pairwise.nonzero()[0])

    # print(a.model_.user_embedding.weight)
    # print(a.model_.item_embedding.weight)

    assert pred[2, 2] > pred[2, 4]


def test_bootstrap_sampling(pageviews_for_pairwise):

    batch_size = 4
    sample_size = 12

    res = bootstrap_sample_pairs(pageviews_for_pairwise, batch_size=batch_size, sample_size=sample_size)

    sample_batch = next(res)

    assert sample_batch.shape[0] == batch_size

    # TODO Add a test to see if it can handle sample_sizes and batch_sizes that are not multiples of eachother.


# def test_samples(pageviews_for_pairwise):
#     a = BPRMF(num_components=2)

#     samples = list(a._generate_samples(pageviews_for_pairwise))
#     assert len(samples) == pageviews_for_pairwise.nnz

#     u = [u for u, _, _ in samples]
#     assert sorted(u) == sorted(pageviews_for_pairwise.nonzero()[0])
