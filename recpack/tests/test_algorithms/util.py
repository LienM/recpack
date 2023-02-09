# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import torch
import torch.optim


def assert_changed(params_before, params_after, device):
    # check if variables have changed
    for (_, p0), (_, p1) in zip(params_before, params_after):
        assert not torch.equal(p0.to(device), p1.to(device))


def assert_same(params_before, params_after, device):
    # check if variables have changed
    for (_, p0), (_, p1) in zip(params_before, params_after):
        assert torch.equal(p0.to(device), p1.to(device))
