# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pandas as pd
import pytest

from recpack.algorithms import SequentialRules
from recpack.matrix import InteractionMatrix


@pytest.fixture()
def algorithm():
    return SequentialRules(K=4)


@pytest.fixture()
def algorithm_k2():
    return SequentialRules(K=2)


@pytest.fixture()
def algorithm_mc():
    """Acts as a markov chain, only considering the previous items when computing cooc."""
    return SequentialRules(K=4, max_steps=1)


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture()
def long_user_histories():
    data = {
        TIMESTAMP_IX: [3, 2, 1, 4, 0, 1, 2, 4],
        ITEM_IX: [0, 1, 2, 3, 0, 1, 2, 4],
        USER_IX: [0, 0, 0, 0, 1, 1, 1, 1],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


def test_fit_1(algorithm, long_user_histories):
    algorithm.fit(long_user_histories)

    # fmt: off
    expected_similarities = np.array([
        [0, 1/2, 1/4, 1/2, 1/6],
        [1/2, 0, 1/2, 1/4, 1/4],
        [1/4, 1/2, 0, 1/6, 1/2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    # fmt: on

    np.testing.assert_array_almost_equal(algorithm.similarity_matrix_.toarray(), expected_similarities)


def test_fit_2(algorithm_k2, long_user_histories):
    algorithm_k2.fit(long_user_histories)

    # fmt: off
    expected_similarities = np.array([
        [0, 1/2, 0, 1/2, 0],
        [1/2, 0, 1/2, 0, 0],
        [0, 1/2, 0, 0, 1/2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    # fmt: on

    np.testing.assert_array_almost_equal(algorithm_k2.similarity_matrix_.toarray(), expected_similarities)


def test_fit_3(algorithm_mc, long_user_histories):
    algorithm_mc.fit(long_user_histories)

    # fmt: off
    expected_similarities = np.array([
        [0, 1/2, 0, 1/2, 0],
        [1/2, 0, 1/2, 0, 0],
        [0, 1/2, 0, 0, 1/2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    # fmt: on

    np.testing.assert_array_almost_equal(algorithm_mc.similarity_matrix_.toarray(), expected_similarities)


def test_predict(algorithm, long_user_histories):
    algorithm.fit(long_user_histories)

    # All histories end in items with no subsequent items in historic sessions
    pred = algorithm.predict(long_user_histories)
    np.testing.assert_array_equal(pred.toarray(), np.zeros(pred.shape))

    # cut off the last items in the histories
    pred_2 = algorithm.predict(long_user_histories.timestamps_lt(4))
    np.testing.assert_array_equal(pred_2.toarray(), algorithm.similarity_matrix_[[0, 2], :].toarray())


def test_no_self_similarity(algorithm, long_user_histories):
    algorithm.fit(long_user_histories)
    n_items = long_user_histories.shape[1]
    assert (algorithm.similarity_matrix_.toarray()[np.arange(n_items, n_items)] == 0).all()
