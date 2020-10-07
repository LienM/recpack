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


@pytest.mark.parametrize("seed", list(range(1, 50)))
def test_pairwise_ranking(pageviews_for_pairwise, seed):
    """Tests that the pairwise ranking of 2 items is correctly computed."""

    # TODO Don't run 50 epochs in a test
    a = BPRMF(num_components=4, num_epochs=3, sample_size=500, batch_size=1, seed=seed)

    a.fit(pageviews_for_pairwise)
    pred = a.predict(pageviews_for_pairwise)

    # # Negative example scores should be lower than positive
    assert pred[1, 2] > pred[1, 4]
    assert pred[1, 1] > pred[1, 4]
    assert pred[1, 0] > pred[1, 4]
    assert pred[1, 2] > pred[1, 3]
    assert pred[1, 1] > pred[1, 3]
    assert pred[1, 0] > pred[1, 3]

    assert pred[3, 3] > pred[3, 0]
    assert pred[3, 4] > pred[3, 0]
    assert pred[3, 3] > pred[3, 1]
    assert pred[3, 4] > pred[3, 1]


def test_bootstrap_sampling(pageviews_for_pairwise):

    batch_size = 4
    sample_size = 12

    res = bootstrap_sample_pairs(
        pageviews_for_pairwise, batch_size=batch_size, sample_size=sample_size
    )

    sample_batch = next(res)

    assert sample_batch.shape[0] == batch_size

    # TODO Add a test to see if it can handle sample_sizes and batch_sizes that are not multiples of eachother.


# def test_samples(pageviews_for_pairwise):
#     a = BPRMF(num_components=2)

#     samples = list(a._generate_samples(pageviews_for_pairwise))
#     assert len(samples) == pageviews_for_pairwise.nnz

#     u = [u for u, _, _ in samples]
#     assert sorted(u) == sorted(pageviews_for_pairwise.nonzero()[0])
