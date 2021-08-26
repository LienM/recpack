import warnings

import numpy as np
import pytest
import scipy.sparse


from recpack.algorithms.base import Algorithm
from recpack.algorithms import (
    ItemKNN, MultVAE, RecVAE, BPRMF, Random,
    NMFItemToItem, NMF, GRU4Rec, Prod2Vec, Prod2VecClustered, ItemPNN)


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


@pytest.mark.parametrize("algo", [ItemPNN, RecVAE, MultVAE, BPRMF, Random,
                                  NMFItemToItem, NMF, GRU4Rec, Prod2Vec, Prod2VecClustered])
def test_seed_is_set_consistently_None(algo):

    a = algo()
    assert hasattr(a, "seed")


@pytest.mark.parametrize("algo", [ItemPNN, RecVAE, MultVAE, BPRMF, Random,
                                  NMFItemToItem, NMF, GRU4Rec, Prod2Vec, Prod2VecClustered])
def test_seed_is_set_consistently_42(algo):

    a = algo(seed=42)
    assert hasattr(a, "seed")

    assert a.seed == 42
