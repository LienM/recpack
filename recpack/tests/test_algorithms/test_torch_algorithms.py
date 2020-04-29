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

# Inspiration for these tests came from:
# https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765

INPUT_SIZE = 1000

torch.manual_seed(400)

inputs = Variable(torch.randn(INPUT_SIZE, INPUT_SIZE))
targets = Variable(torch.randint(0, 1, (INPUT_SIZE,))).long()
batch = [inputs, targets]


@pytest.fixture(scope="function")
def mult_vae():
    return MultVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        max_beta=0.2,
        anneal_steps=2,
        dropout=0.5,
    )


def test_training_epoch():
    pass


def test_evaluation_epoch():
    pass


def test_multi_vae_forward():
    model = MultiVAETorch(dim_input_layer=INPUT_SIZE)
    # do they change after a training step?
    #  let's run a train step and see
    parameters = model.named_parameters()

    vars_to_change = [
        (param_name, param)
        for param_name, param in parameters
        if param_name not in ("p_bn_hid_layer.weight", "p_bn_hid_layer.bias", "p_hid_out_layer.weight", "p_hid_out_layer.bias")
    ]

    assert_vars_change(
        model,
        vae_loss_function,
        torch.optim.Adam(model.parameters()),
        batch,
        "cpu",
        params=vars_to_change,
    )


def test_multi_vae_backup_forward():
    model = MultiVAE([200, 600, INPUT_SIZE])
    # do they change after a training step?
    #  let's run a train step and see
    parameters = model.named_parameters()

    vars_to_change = [
        (param_name, param)
        for param_name, param in parameters
        if param_name not in ("p_layers.0.weight", "p_layers.0.bias", "p_layers.1.weight", "p_layers.1.bias")
    ]

    assert_vars_change(
        model,
        vae_loss_function,
        torch.optim.Adam(model.parameters()),
        batch,
        "cpu",
        params=vars_to_change,
    )
