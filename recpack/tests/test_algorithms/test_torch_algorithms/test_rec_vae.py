
from unittest.mock import MagicMock
from typing import Callable

import pytest
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from recpack.algorithms.torch_algorithms.rec_vae import (
    RecVAETorch,
    RecVAE
)

from utils import assert_changed, assert_same

@pytest.fixture(scope="function")
def rec_vae():
    mult = RecVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        dropout=0.5,
    )

    mult.save = MagicMock(return_value=True)

    return mult


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
