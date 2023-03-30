# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.algorithms import TARSItemKNNHermann
import pandas as pd
import numpy as np
import pytest
from recpack.matrix import InteractionMatrix


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def mat():
    data = {
        TIMESTAMP_IX: [1, 1, 1, 2, 3, 4],
        ITEM_IX: [0, 0, 1, 2, 1, 2],
        USER_IX: [0, 1, 2, 2, 1, 0],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX, shape=(3, 4))


@pytest.fixture(scope="function")
def mat_no_timestamps():
    data = {
        TIMESTAMP_IX: [0, 0, 0, 0, 0, 0],
        ITEM_IX: [0, 0, 1, 2, 1, 2],
        USER_IX: [0, 1, 2, 2, 1, 0],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture()
def algorithm():
    return TARSItemKNNHermann(K=2)


@pytest.fixture()
def algorithm_1h_interval():
    return TARSItemKNNHermann(K=2, decay_interval=3600)


def test_time_decay_knn_fit(mat, algorithm):
    algorithm.fit(mat)
    # fmt: off
    expected_similarities = np.array(
        [
            [0, 1 / 6, 1 / 7, 0],
            [1 / 6, 0, 1 / 5, 0],
            [1 / 7, 1 / 5, 0, 0],
            [0, 0, 0, 0]
        ]
    )
    # fmt: on

    np.testing.assert_almost_equal(algorithm.similarity_matrix_.toarray(), expected_similarities)


def test_time_decay_knn_fit_configured_time_unit(mat, algorithm_1h_interval):
    algorithm_1h_interval.fit(mat)

    # fmt: off
    expected_similarities = np.array(
        [
            [0, 3600 / 6, 3600 / 7, 0],
            [3600 / 6, 0, 3600 / 5, 0],
            [3600 / 7, 3600 / 5, 0, 0],
            [0, 0, 0, 0]
        ]
    )
    # fmt: on

    np.testing.assert_almost_equal(algorithm_1h_interval.similarity_matrix_.toarray(), expected_similarities)


def test_time_decay_knn_predict(mat, algorithm):
    algorithm.fit(mat)
    pred = algorithm.predict(mat)

    assert pred.shape == mat.shape
