
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


def _training_step(model: nn.Module, optim: torch.optim.Optimizer, inputs: Variable, targets: Variable, device: torch.device):

    # put model in train mode
    model.train()
    model.to(device)

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()
    # move data to device
    inputs = inputs.to(device)
    targets = targets.to(device)
    # forward
    _, loss = model(inputs)
    # backward
    loss.backward()
    # optimization step
    optim.step()


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


def test_multi_vae_forward(input_size, inputs, targets):
    rec_vae = RecVAETorch(600, 200, dim_input_layer=input_size)

    params = [np for np in rec_vae.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    device = torch.device("cpu")

    _training_step(rec_vae, torch.optim.Adam(rec_vae.parameters()), inputs, targets, device)
    # do they change after a training step?
    #  let's run a train step and see

    assert_changed(params_before, params, device)
