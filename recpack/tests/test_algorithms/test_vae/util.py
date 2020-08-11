import torch
from typing import Callable
import torch.nn as nn
import torch.optim
from torch.autograd import Variable


def assert_changed(params_before, params_after, device):
    # check if variables have changed
    for (_, p0), (_, p1) in zip(params_before, params_after):
        assert not torch.equal(p0.to(device), p1.to(device))


def assert_same(params_before, params_after, device):
    # check if variables have changed
    for (_, p0), (_, p1) in zip(params_before, params_after):
        assert torch.equal(p0.to(device), p1.to(device))
