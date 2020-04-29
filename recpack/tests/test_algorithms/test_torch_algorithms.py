import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from recpack.algorithms.torch_algorithms.vaes import (
    MultiVAETorch,
    MultVAE,
    vae_loss_function,
)

from recpack.algorithms.torch_algorithms.vaes_backup import (
    MultiVAE,
    loss_function,
)

from recpack.tests.test_algorithms.torch_test_helpers import assert_vars_change

INPUT_SIZE = 1000

inputs = Variable(torch.randn(INPUT_SIZE, INPUT_SIZE))
targets = Variable(torch.randint(0, 1, (INPUT_SIZE,))).long()
batch = [inputs, targets]


def test_training_epoch():
    pass


def test_evaluation_epoch():
    pass


def test_multi_vae_forward():
    model = MultiVAETorch(dim_input_layer=INPUT_SIZE)
    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model, vae_loss_function, torch.optim.Adam(model.parameters()), batch, "cpu"
    )


def test_multi_vae_backup_forward():
    model = MultiVAE([200, 600, INPUT_SIZE])
    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model, loss_function, torch.optim.Adam(model.parameters()), batch, "cpu"
    )
