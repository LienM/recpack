from typing import Callable

import scipy.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from recpack.algorithms.vae.mult_vae import (
    MultiVAETorch,
    vae_loss_function,
)
from recpack.tests.test_algorithms.util import assert_changed, assert_same


def _training_step(
    model: nn.Module,
    loss_fn: Callable,
    optim: torch.optim.Optimizer,
    inputs: Variable,
    targets: Variable,
    device: torch.device
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
    mult_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    params = [np for np in mult_vae.model_.named_parameters()
              if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    mult_vae._train_epoch(larger_matrix, users)

    device = mult_vae.device

    assert_changed(params_before, params, device)


def test_evaluation_epoch(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    params = [np for np in mult_vae.model_.named_parameters()
              if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    mult_vae._evaluate(larger_matrix, larger_matrix, users)

    device = mult_vae.device

    assert_same(params_before, params, device)


def test_predict(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix.shape[1])

    X_pred = mult_vae.predict(larger_matrix)

    assert isinstance(X_pred, scipy.sparse.csr_matrix)

    assert not set(X_pred.nonzero()[0]).difference(larger_matrix.nonzero()[0])


def test_multi_vae_forward(input_size, inputs, targets):
    mult_vae = MultiVAETorch(dim_input_layer=input_size)

    params = [np for np in mult_vae.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    device = torch.device("cpu")

    _training_step(mult_vae, vae_loss_function, torch.optim.Adam(
        mult_vae.parameters()), inputs, targets, device)
    # do they change after a training step?
    #  let's run a train step and see

    assert_changed(params_before, params, device)
