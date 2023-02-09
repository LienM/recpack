# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import scipy.sparse
import tempfile
from unittest.mock import MagicMock


from recpack.tests.test_algorithms.util import assert_changed, assert_same

from recpack.algorithms import RecVAE


def rec_vae():
    rec = RecVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        dropout=0.5,
    )

    rec.save = MagicMock(return_value=True)

    return rec


def rec_vae_beta():
    rec = RecVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        dropout=0.5,
        gamma=None,
        beta=0.05,
    )

    rec.save = MagicMock(return_value=True)

    return rec


@pytest.mark.parametrize("algo", [rec_vae(), rec_vae_beta()])
def test_encoder_training(larger_matrix, algo):
    algo._init_model(larger_matrix)

    users = list(set(larger_matrix.nonzero()[1]))
    encoder_params = [np for np in algo.model_.encoder.named_parameters() if np[1].requires_grad]
    decoder_params = [np for np in algo.model_.decoder.named_parameters() if np[1].requires_grad]

    # take a copy
    encoder_params_before = [(name, p.clone()) for (name, p) in encoder_params]
    decoder_params_before = [(name, p.clone()) for (name, p) in decoder_params]

    algo.best_model = tempfile.NamedTemporaryFile()
    # run a training step
    algo._train_partial(larger_matrix, users, algo.enc_optimizer)

    device = algo.device

    assert_changed(encoder_params_before, encoder_params, device)
    assert_same(decoder_params_before, decoder_params, device)

    algo.best_model.close()


@pytest.mark.parametrize("algo", [rec_vae(), rec_vae_beta()])
def test_decoder_training(larger_matrix, algo):
    algo._init_model(larger_matrix)

    users = list(set(larger_matrix.nonzero()[1]))
    encoder_params = [np for np in algo.model_.encoder.named_parameters() if np[1].requires_grad]
    decoder_params = [np for np in algo.model_.decoder.named_parameters() if np[1].requires_grad]

    # take a copy
    encoder_params_before = [(name, p.clone()) for (name, p) in encoder_params]
    decoder_params_before = [(name, p.clone()) for (name, p) in decoder_params]

    # run a training step
    algo._train_partial(larger_matrix, users, algo.dec_optimizer)

    device = algo.device

    assert_same(encoder_params_before, encoder_params, device)
    assert_changed(decoder_params_before, decoder_params, device)


@pytest.mark.parametrize("algo", [rec_vae(), rec_vae_beta()])
def test_training_epoch(larger_matrix, algo):
    algo._init_model(larger_matrix)

    params = [np for np in algo.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    algo._train_epoch(larger_matrix)

    device = algo.device

    assert_changed(params_before, params, device)


@pytest.mark.parametrize("algo", [rec_vae(), rec_vae_beta()])
def test_evaluation_epoch(larger_matrix, algo):
    algo._init_model(larger_matrix)

    params = [np for np in algo.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    algo.best_model = tempfile.NamedTemporaryFile()
    # run a training step
    algo._evaluate(larger_matrix, larger_matrix)

    device = algo.device

    assert_same(params_before, params, device)
    algo.best_model.close()


@pytest.mark.parametrize("algo", [rec_vae(), rec_vae_beta()])
def test_predict(larger_matrix, algo):
    algo._init_model(larger_matrix)

    X_pred = algo.predict(larger_matrix)

    assert isinstance(X_pred, scipy.sparse.csr_matrix)

    assert not set(X_pred.nonzero()[0]).difference(larger_matrix.nonzero()[0])
