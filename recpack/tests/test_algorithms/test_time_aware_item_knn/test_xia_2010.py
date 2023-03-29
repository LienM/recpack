# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


from recpack.algorithms.time_aware_item_knn import (
    TARSItemKNNXia,
)
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

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture(scope="function")
def mat_no_timestamps():
    data = {
        ITEM_IX: [0, 0, 1, 2, 1, 2],
        USER_IX: [0, 1, 2, 2, 1, 0],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX)


@pytest.fixture()
def mat_diag():
    # Make can use the diag matrix to make sure predict works on 1 item histories.
    # If we create users with a single item seen in order.
    data = {
        TIMESTAMP_IX: [1, 1, 1],
        ITEM_IX: [0, 1, 2],
        USER_IX: [0, 1, 2],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture()
def mat_1_user_2_visits():
    # Make can use the diag matrix to make sure predict works on 1 item histories.
    # If we create users with a single item seen in order.
    data = {
        TIMESTAMP_IX: [1, 2],
        ITEM_IX: [0, 1],
        USER_IX: [0, 0],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX, shape=(1, 3))


@pytest.mark.parametrize(
    "decay_function, fit_decay, expected_similarities, expected_out",
    [
        (
            "linear",
            0.5,
            np.array([[0.0, 2 / 3, 1 / 2], [2 / 3, 0.0, 5 / 6], [1 / 2, 5 / 6, 0.0]]),
            [[2 / 3, 2 / 3, 4 / 3]],
        ),
        (
            "convex",
            0.5,
            np.array([[0.0, 1 / 4, 1 / 8], [1 / 4, 0.0, 1 / 2], [1 / 8, 1 / 2, 0.0]]),
            [[1 / 4, 1 / 4, 5 / 8]],
        ),
        (
            "concave",
            0.5,
            np.array(
                [
                    [0.0, 1 - (0.5 ** (1 / 3)), 0.0],
                    [1 - (0.5 ** (1 / 3)), 0.0, 1 - (0.5 ** (2 / 3))],
                    [0.0, 1 - (0.5 ** (2 / 3)), 0.0],
                ]
            ),
            [[1 - (0.5 ** (1 / 3)), 1 - (0.5 ** (1 / 3)), 1 - (0.5 ** (2 / 3))]],
        ),
    ],
)
def test_time_decay_knn(
    mat, mat_diag, mat_1_user_2_visits, decay_function, fit_decay, expected_similarities, expected_out
):
    algo = TARSItemKNNXia(K=2, fit_decay=fit_decay, decay_function=decay_function, decay_interval=1)

    algo.fit(mat)

    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)

    result = algo.predict(mat_diag)
    np.testing.assert_almost_equal(result.toarray(), expected_similarities)

    result = algo.predict(mat_1_user_2_visits)
    np.testing.assert_almost_equal(result.toarray(), expected_out)


@pytest.mark.parametrize(
    "decay_function, fit_decay",
    [
        (
            "convex",
            0.5,
        ),
    ],
)
def test_time_decay_knn_empty(mat_no_timestamps, decay_function, fit_decay):
    algo = TARSItemKNNXia(K=2, fit_decay=fit_decay, decay_function=decay_function, decay_interval=1)

    with pytest.raises(ValueError) as record:
        algo.fit(mat_no_timestamps)


@pytest.mark.parametrize(
    "decay_function, fit_decay",
    [
        ("linear", -0.1),
        ("concave", 1.1),
        ("concave", -0.1),
        ("convex", 2),
    ],
)
def test_time_decay_knn_coeff_validation(decay_function, fit_decay):
    with pytest.raises(ValueError):
        TARSItemKNNXia(fit_decay=fit_decay, decay_function=decay_function)


@pytest.mark.parametrize(
    "decay_function",
    [
        "linearity",
        "cosine",
        "crap",
    ],
)
def test_time_decay_knn_fn_validation_error(decay_function):
    with pytest.raises(ValueError):
        TARSItemKNNXia(decay_function=decay_function)


@pytest.mark.parametrize(
    "decay_function, fit_decay",
    [
        (
            "linear",
            0.5,
        ),
    ],
)
def test_time_decay_knn_predict(mat, mat_diag, decay_function, fit_decay):
    algo = TARSItemKNNXia(K=2, fit_decay=fit_decay, decay_function=decay_function, decay_interval=1)

    algo.fit(mat)
    assert isinstance(algo.similarity_matrix_, csr_matrix)

    result = algo.predict(mat_diag)
    assert isinstance(result, csr_matrix)

    np.testing.assert_array_equal(result.toarray(), algo.similarity_matrix_.toarray())


@pytest.mark.parametrize("decay_interval", [0, 0.5])
def test_decay_interval_validation(decay_interval):
    with pytest.raises(ValueError):
        TARSItemKNNXia(decay_interval=decay_interval)
