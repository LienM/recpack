# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

from recpack.algorithms import BPRMF
from recpack.algorithms.bprmf import MFModule


@pytest.fixture(scope="function")
def X_in_for_pairwise():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 1, 1, 1, 3, 3, 4, 4],
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    pv = csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


def test_bprmf(X_in):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1)
    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])


def test_bprmf_topK(X_in):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1, predict_topK=1)

    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])
    # Each user should receive a single recommendation
    assert pred.nonzero()[1].shape[0] == len(set(X_in.nonzero()[0]))


def test_bprmf_w_interaction_mat(X_in_interaction_m):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1)
    a.fit(X_in_interaction_m, (X_in_interaction_m, X_in_interaction_m))

    pred = a.predict(X_in_interaction_m)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(X_in_interaction_m.active_users)


@pytest.mark.parametrize("seed", list(range(1, 25)))
def test_pairwise_ranking(X_in_for_pairwise, seed):
    """Tests that the pairwise ranking of 2 items is correctly computed."""

    a = BPRMF(
        num_components=4,
        max_epochs=10,
        batch_size=10,
        seed=seed,
        learning_rate=0.5,
        sample_size=50,
    )
    a.fit(X_in_for_pairwise, (X_in_for_pairwise, X_in_for_pairwise))
    pred = a.predict(X_in_for_pairwise)

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


def test_save_and_load(X_in_for_pairwise):
    a = BPRMF(
        num_components=4,
        max_epochs=1,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    a.fit(X_in_for_pairwise, (X_in_for_pairwise, X_in_for_pairwise))

    assert os.path.isfile(a.filename)

    b = BPRMF(
        num_components=4,
        max_epochs=40,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    b.load(a.filename)

    np.testing.assert_array_equal(
        a.predict(X_in_for_pairwise).toarray(),
        b.predict(X_in_for_pairwise).toarray(),
    )

    # TODO cleanup
    os.remove(a.filename)


def test_forward(X_in_for_pairwise):
    a = MFModule(3, 3, 2)

    U = torch.LongTensor([0, 1])
    I = torch.LongTensor([0, 2])

    res_1 = a.forward(U, I)
    res_2 = a.forward(U[1], I[1])

    assert res_2 == res_1[1, 1]


def test_bad_stopping_criterion(X_in):
    with pytest.raises(ValueError):
        BPRMF(stopping_criterion="not_a_correct_value")


def test_recall_stopping_criterion(X_in):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1, stopping_criterion="recall")
    a.fit(X_in, (X_in, X_in))
