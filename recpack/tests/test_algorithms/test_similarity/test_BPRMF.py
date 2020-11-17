from functools import partial
import os

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from recpack.algorithms import BPRMF
from recpack.algorithms.similarity.BPRMF import MFModule, bootstrap_sample_pairs
from recpack.algorithms.util import StoppingCriterion
from recpack.metrics.recall import recall_k


def test_bprmf(pageviews):
    a = BPRMF(num_components=2, num_epochs=2, batch_size=1)
    a.fit(pageviews, (pageviews, pageviews))

    pred = a.predict(pageviews)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(pageviews.nonzero()[0])


@pytest.mark.parametrize("seed", list(range(1, 25)))
def test_pairwise_ranking(pageviews_for_pairwise, seed):
    """Tests that the pairwise ranking of 2 items is correctly computed."""

    a = BPRMF(
        num_components=4,
        num_epochs=40,
        batch_size=2,
        seed=seed,
        learning_rate=0.05,
    )
    a.fit(pageviews_for_pairwise, (pageviews_for_pairwise, pageviews_for_pairwise))
    pred = a.predict(pageviews_for_pairwise)

    # Negative example scores should be lower than positive
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


def test_save_and_load(pageviews_for_pairwise):
    a = BPRMF(
        num_components=4,
        num_epochs=40,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    a.fit(pageviews_for_pairwise, (pageviews_for_pairwise, pageviews_for_pairwise))

    assert os.path.isfile(a.file_name)

    b = BPRMF(
        num_components=4,
        num_epochs=40,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    b.load(a.file_name)

    np.testing.assert_array_equal(
        a.predict(pageviews_for_pairwise).toarray(),
        b.predict(pageviews_for_pairwise).toarray(),
    )

    # TODO cleanup
    os.remove(a.file_name)


@pytest.mark.parametrize(
    "batch_size, sample_size",
    [
        (4, 12),
        (3, 10),
    ],
)
def test_bootstrap_sampling(pageviews_for_pairwise, batch_size, sample_size):

    res = bootstrap_sample_pairs(
        pageviews_for_pairwise, batch_size=batch_size, sample_size=sample_size
    )

    sample_batch = next(res)

    assert sample_batch.shape[0] == batch_size
    b = 0
    for batch in res:
        b = batch.shape[0]

    target_size = sample_size % batch_size
    if target_size == 0:
        # Special case where it matches.
        target_size = batch_size

    assert b == target_size


def test_forward(pageviews_for_pairwise):
    a = MFModule(3, 3, 2)

    U = torch.LongTensor([0, 1])
    I = torch.LongTensor([0, 2])

    res_1 = a.forward(U, I)
    res_2 = a.forward(U[1], I[1])

    assert res_2 == res_1[1, 1]


def test_bad_stopping_criterion(pageviews):
    with pytest.raises(RuntimeError):
        BPRMF(stopping_criterion="not_a_correct_value")


def test_recall_stopping_criterion(pageviews):

    a = BPRMF(num_components=2, num_epochs=2, batch_size=1, stopping_criterion="recall")
    a.fit(pageviews, (pageviews, pageviews))
