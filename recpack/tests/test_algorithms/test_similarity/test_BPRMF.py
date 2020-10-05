import pytest
import sklearn
from recpack.algorithms.similarity.BPRMF import BPRMF


def test_bprmf(pageviews):
    a = BPRMF(num_components=2)
    a.fit(pageviews)

    a2 = BPRMF(num_components=2, reg=0.01)
    a2.fit(pageviews)

    print(a.model_.user_embedding.weight)

    pred = a.predict(pageviews)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(pageviews.nonzero()[0])


def test_pairwise_ranking(pageviews_for_pairwise):
    """Tests that the pairwise ranking of 2 items is correctly computed."""

    from recpack.algorithms import SVD

    b = SVD(2)
    b.fit(pageviews_for_pairwise)
    pred = b.predict(pageviews_for_pairwise)

    assert pred[2, 2] > pred[2, 4]

    a = BPRMF(num_components=2, reg=0.0001, learning_rate=0.0001, num_epochs=50)

    a.fit(pageviews_for_pairwise)
    print(pageviews_for_pairwise.toarray())
    pred = a.predict(pageviews_for_pairwise)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(pageviews_for_pairwise.nonzero()[0])

    print(a.model_.user_embedding.weight)
    print(a.model_.item_embedding.weight)

    assert pred[2, 2] > pred[2, 4]


def test_samples(pageviews_for_pairwise):
    a = BPRMF(num_components=2)

    samples = list(a._generate_samples(pageviews_for_pairwise))
    assert len(samples) == pageviews_for_pairwise.nnz

    u = [u for u, _, _ in samples]
    assert sorted(u) == sorted(pageviews_for_pairwise.nonzero()[0])
