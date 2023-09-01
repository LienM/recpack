# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import warnings

import numpy as np
import pytest
import scipy.sparse
from unittest.mock import MagicMock


from recpack.algorithms.base import Algorithm
from recpack.algorithms import (
    ItemKNN,
    MultVAE,
    RecVAE,
    BPRMF,
    Random,
    NMFItemToItem,
    NMF,
    GRU4RecCrossEntropy,
    GRU4RecNegSampling,
    Prod2Vec,
    Prod2VecClustered,
    ItemPNN,
)


def test_check_prediction():
    a = np.ones(5)
    b = a.copy()
    b[2] = 0
    X_pred = scipy.sparse.diags(b).tocsr()
    X = scipy.sparse.diags(a).tocsr()

    a = Algorithm()
    
    with pytest.warns(UserWarning, match="1 users") as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a._check_prediction(X_pred, X)

    with pytest.warns(None):
        a._check_prediction(X, X)


def test_check_fit_complete(X_in):
    # Set a row to 0, so it won't have any neighbours
    pv_copy = X_in.copy()
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
    [
        ItemPNN,
        RecVAE,
        MultVAE,
        BPRMF,
        Random,
        NMFItemToItem,
        NMF,
        Prod2Vec,
        Prod2VecClustered,
        GRU4RecCrossEntropy,
        GRU4RecNegSampling,
    ],
)
def test_seed_is_set_consistently_None(algo):

    a = algo()
    assert hasattr(a, "seed")


@pytest.mark.parametrize(
    "algo",
    [
        ItemPNN,
        RecVAE,
        MultVAE,
        BPRMF,
        Random,
        NMFItemToItem,
        NMF,
        Prod2Vec,
        Prod2VecClustered,
        GRU4RecNegSampling,
        GRU4RecCrossEntropy,
    ],
)
def test_seed_is_set_consistently_42(algo):

    a = algo(seed=42)
    assert hasattr(a, "seed")

    assert a.seed == 42


@pytest.mark.parametrize(
    "algo_class",
    [RecVAE, MultVAE, BPRMF, Prod2Vec, Prod2VecClustered, GRU4RecNegSampling, GRU4RecCrossEntropy],
)
def test_assert_is_interaction_matrix(algo_class, matrix_sessions):
    # No error when checking type
    algo = algo_class()

    algo._assert_is_interaction_matrix(matrix_sessions)

    with pytest.raises(TypeError) as type_error:
        algo._assert_is_interaction_matrix(matrix_sessions.binary_values)

    assert type_error.match(".* requires Interaction Matrix as input. Got <class 'scipy.sparse._csr.csr_matrix'>.")


@pytest.mark.parametrize(
    "algo_class",
    [RecVAE, MultVAE, BPRMF, Prod2Vec, Prod2VecClustered, GRU4RecNegSampling, GRU4RecCrossEntropy],
)
def test_assert_has_timestamps(algo_class, matrix_sessions):

    algo = algo_class()

    with pytest.raises(ValueError) as value_error:
        algo._assert_has_timestamps(matrix_sessions.eliminate_timestamps())

    assert value_error.match(".* requires timestamp information in the InteractionMatrix.")


@pytest.mark.parametrize(
    "algo_class",
    [RecVAE, MultVAE, BPRMF, Prod2Vec, Prod2VecClustered, GRU4RecNegSampling, GRU4RecCrossEntropy],
)
def test_sampled_validation(algo_class, larger_mat):
    N_SAMPLES = 50
    algo = algo_class(validation_sample_size=N_SAMPLES, max_epochs=5)

    # We will mock the update function, so we can check it's input has N_SAMPLES rows,
    # as we expect this behaviour after the sample.
    mock = MagicMock()
    mock.update.return_value = True

    algo.stopping_criterion = mock

    algo.fit(larger_mat, (larger_mat, larger_mat))

    # Make sure that the model was fitted
    assert algo.model_ is not None

    for c in mock.update.call_args_list:
        val_out, X_pred = c.args
        assert len(set(val_out.nonzero()[0])) == N_SAMPLES
        assert len(set(X_pred.nonzero()[0])) == N_SAMPLES
