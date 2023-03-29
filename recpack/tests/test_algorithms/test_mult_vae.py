# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
from typing import Callable
from unittest.mock import MagicMock

import pytest
import scipy.sparse
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from recpack.algorithms import MultVAE
from recpack.algorithms.loss_functions import vae_loss
from recpack.algorithms.mult_vae import (
    MultiVAETorch,
)
from recpack.tests.test_algorithms.util import assert_changed, assert_same

# Inspiration for these tests came from:
# https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765


@pytest.fixture(scope="function")
def mult_vae():
    mult = MultVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        max_beta=0.2,
        anneal_steps=20,
        dropout=0.5,
    )

    mult.save = MagicMock(return_value=True)

    return mult


def _training_step(
    model: nn.Module,
    loss_fn: Callable,
    optim: optim.Optimizer,
    inputs: Variable,
    targets: Variable,
    device: torch.device,
):
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
    likelihood = model(inputs)
    # calc loss
    loss = loss_fn(*likelihood, targets)
    # backward
    loss.backward()
    # optimization step
    optim.step()


def test_training_epoch(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix)

    params = [np for np in mult_vae.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    mult_vae._train_epoch(larger_matrix)

    device = mult_vae.device

    assert_changed(params_before, params, device)


def test_evaluation_epoch(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix)

    params = [np for np in mult_vae.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    mult_vae.best_model = tempfile.NamedTemporaryFile()
    # run a training step
    mult_vae._evaluate(larger_matrix, larger_matrix)

    device = mult_vae.device

    assert_same(params_before, params, device)
    mult_vae.best_model.close()


def test_predict(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix)

    X_pred = mult_vae.predict(larger_matrix)

    assert isinstance(X_pred, scipy.sparse.csr_matrix)

    assert not set(X_pred.nonzero()[0]).difference(larger_matrix.nonzero()[0])


def test_multi_vae_forward():
    input_size = 100

    torch.manual_seed(400)
    inputs = Variable(torch.randn(input_size, input_size))
    targets = Variable(torch.randint(0, 2, (input_size,))).long()

    mult_vae = MultiVAETorch(dim_input_layer=input_size)

    params = [np for np in mult_vae.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    device = torch.device("cpu")

    _training_step(
        mult_vae,
        vae_loss,
        optim.Adam(mult_vae.parameters()),
        inputs,
        targets,
        device,
    )
    # do they change after a training step?
    #  let's run a train step and see

    assert_changed(params_before, params, device)
