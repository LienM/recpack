
from unittest.mock import MagicMock
from typing import Callable

import pytest
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from recpack.algorithms.vae.rec_vae import (
    RecVAETorch,
    RecVAE
)

from util import assert_changed, assert_same


def test_encoder_training(rec_vae, larger_matrix):
    rec_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    encoder_params = [np for np in rec_vae.model_.encoder.named_parameters() if np[1].requires_grad]
    decoder_params = [np for np in rec_vae.model_.decoder.named_parameters() if np[1].requires_grad]

    # take a copy
    encoder_params_before = [(name, p.clone()) for (name, p) in encoder_params]
    decoder_params_before = [(name, p.clone()) for (name, p) in decoder_params]
    
    # run a training step
    rec_vae._train_encoder_or_decoder(larger_matrix, users, train_encoder=True)

    device = rec_vae.device

    assert_changed(encoder_params_before, encoder_params, device)
    assert_same(decoder_params_before, decoder_params, device)

def test_decoder_training(rec_vae, larger_matrix):
    rec_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    encoder_params = [np for np in rec_vae.model_.encoder.named_parameters() if np[1].requires_grad]
    decoder_params = [np for np in rec_vae.model_.decoder.named_parameters() if np[1].requires_grad]

    # take a copy
    encoder_params_before = [(name, p.clone()) for (name, p) in encoder_params]
    decoder_params_before = [(name, p.clone()) for (name, p) in decoder_params]
    
    # run a training step
    rec_vae._train_encoder_or_decoder(larger_matrix, users, train_encoder=False)

    device = rec_vae.device

    assert_same(encoder_params_before, encoder_params, device)
    assert_changed(decoder_params_before, decoder_params, device)


def test_training_epoch(rec_vae, larger_matrix):
    rec_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    params = [np for np in rec_vae.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    rec_vae._train_epoch(larger_matrix, users)

    device = rec_vae.device

    assert_changed(params_before, params, device)


def test_evaluation_epoch(rec_vae, larger_matrix):
    rec_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    params = [np for np in rec_vae.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    rec_vae._evaluate(larger_matrix, larger_matrix, users)

    device = rec_vae.device

    assert_same(params_before, params, device)


def test_predict(rec_vae, larger_matrix):
    rec_vae._init_model(larger_matrix.shape[1])

    X_pred = rec_vae.predict(larger_matrix)

    assert isinstance(X_pred, scipy.sparse.csr_matrix)

    assert not set(X_pred.nonzero()[0]).difference(larger_matrix.nonzero()[0])
