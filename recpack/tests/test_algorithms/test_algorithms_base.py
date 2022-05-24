import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse


from recpack.algorithms.base import Algorithm
from recpack.algorithms import (
    ItemKNN,
    MultVAE,
    RecVAE,
    BPRMF,
    Random,
    NMFItemToItem,
    NMF,
    GRU4Rec,
    GRU4RecCrossEntropy,
    GRU4RecNegSampling,
    Prod2Vec,
    Prod2VecClustered,
    ItemPNN,
    CASER,
    REBUS,
)
from recpack.data.matrix import InteractionMatrix


def test_check_prediction():
    a = np.ones(5)
    b = a.copy()
    b[2] = 0
    X_pred = scipy.sparse.diags(b).tocsr()
    X = scipy.sparse.diags(a).tocsr()

    a = Algorithm()

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a._check_prediction(X_pred, X)
        # Verify some things
        assert len(w) == 1

        assert "1 users" in str(w[-1].message)

        a._check_prediction(X, X)
        assert len(w) == 1


def test_check_fit_complete(pageviews):
    # Set a row to 0, so it won't have any neighbours
    pv_copy = pageviews.copy()
    pv_copy[:, 4] = 0

    a = ItemKNN(2)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        a.fit(pv_copy)
        # The algorithm might also throw a warning
        assert len(w) >= 1

        assert "1 items" in str(w[-1].message)


@pytest.mark.parametrize(
    "algo",
    [ItemPNN, RecVAE, MultVAE, BPRMF, Random, NMFItemToItem, NMF, GRU4Rec, Prod2Vec, Prod2VecClustered, CASER, REBUS],
)
def test_seed_is_set_consistently_None(algo):

    a = algo()
    assert hasattr(a, "seed")


@pytest.mark.parametrize(
    "algo",
    [ItemPNN, RecVAE, MultVAE, BPRMF, Random, NMFItemToItem, NMF, GRU4Rec, Prod2Vec, Prod2VecClustered, CASER, REBUS],
)
def test_seed_is_set_consistently_42(algo):

    a = algo(seed=42)
    assert hasattr(a, "seed")

    assert a.seed == 42


@pytest.mark.parametrize(
    "algo",
    [CASER, REBUS],
)
def test_pytorch_num_items_check(algo, non_duplicate_matrix_sessions):
    a = algo(max_epochs=3)
    a.fit(non_duplicate_matrix_sessions, (non_duplicate_matrix_sessions, non_duplicate_matrix_sessions))

    USER_IX = InteractionMatrix.USER_IX
    ITEM_IX = InteractionMatrix.ITEM_IX
    TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX

    df = pd.DataFrame(
        {
            USER_IX: [1, 1, 3, 3, 4],
            ITEM_IX: [1, 2, 1, 2, 2],
            TIMESTAMP_IX: [1, 2, 3, 4, 5],
        }
    )
    im = InteractionMatrix(df, user_ix=USER_IX, item_ix=ITEM_IX, timestamp_ix=TIMESTAMP_IX)

    with pytest.raises(ValueError) as value_error:
        a.predict(im)

    assert value_error.match(
        "Shape mismatch between learned model and prediction data. "
        f"Expected {non_duplicate_matrix_sessions.shape[1]} items, got {im.shape[1]} instead."
    )
