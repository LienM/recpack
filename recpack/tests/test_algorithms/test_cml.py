from typing import Callable
from unittest.mock import MagicMock

import scipy.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytest

from recpack.algorithms.metric_learning.cml import (
    CML,
    CMLTorch,
    warp_loss,
)
from recpack.tests.test_algorithms.util import assert_changed, assert_same

# TODO Add test to check what happens if batch_size > sample_size


@pytest.fixture(scope="function")
def cml():
    cml1 = CML(
        100,  # num_components
        1.9,  # margin
        0.1,  # learning_rate
        1,  # clip_norm
        False,  # use_cov_loss
        2,  # num_epochs
        seed=42,
        batch_size=20,
        U=10,
    )

    cml1.save = MagicMock(return_value=True)

    return cml1


def test_cml_training_epoch(cml, larger_matrix):
    cml._init_model(larger_matrix)

    params = [np for np in cml.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    cml._train_epoch(larger_matrix)

    device = cml.device

    assert_changed(params_before, params, device)


def test_cml_evaluation_epoch(cml, larger_matrix):
    cml._init_model(larger_matrix)

    params = [np for np in cml.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # Expects a tuple: validation_data = (validation_data_in, validation_data_out)
    cml._evaluate((larger_matrix, larger_matrix))

    device = cml.device

    assert_same(params_before, params, device)


def test_cml_predict(cml, larger_matrix):
    cml._init_model(larger_matrix)

    X_pred = cml.predict(larger_matrix)

    assert isinstance(X_pred, scipy.sparse.csr_matrix)

    assert not set(X_pred.nonzero()[0]).difference(larger_matrix.nonzero()[0])
