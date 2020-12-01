import os.path

# from unittest.mock import MagicMock

import numpy as np
import scipy.sparse
import pytest
import torch.nn as nn

from recpack.data.matrix import to_datam
from recpack.splitters.scenarios import StrongGeneralization
from recpack.algorithms.metric_learning.cml import (
    CML,
    CMLTorch,
    # warp_loss,
    covariance_loss,
)
from recpack.tests.test_algorithms.util import assert_changed, assert_same

# TODO Add test to check what happens if batch_size > sample_size


@pytest.fixture(scope="function")
def cml():
    cml1 = CML(
        100,  # num_components
        1.9,  # margin
        0.1,  # learning_rate
        2,  # num_epochs
        seed=42,
        batch_size=20,
        U=10,
    )

    return cml1


@pytest.fixture(scope="function")
def cml_save():
    cml1 = CML(
        100,  # num_components
        1.9,  # margin
        0.1,  # learning_rate
        2,  # num_epochs
        seed=42,
        batch_size=20,
        U=10,
        save_best_to_file=True,
    )

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


def test_cml_training_epoch_w_disentanglement(cml, larger_matrix):
    cml.disentange = True
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


# Test if matrix changed between before and after approximate?
def test_cml_predict_w_approximate(cml, larger_matrix):
    cml.approximate_user_vectors = True

    dm = to_datam(larger_matrix)
    s = StrongGeneralization(0.7, 1.0, validation=True)

    s.split(dm)

    cml.fit(s.training_data, s.validation_data)

    assert cml.known_users_ == set(s.training_data.binary_values.nonzero()[0])

    X_pred = cml.predict(s.test_data_in)

    assert cml.known_users_ == set(s.training_data.binary_values.nonzero()[0])

    W_as_tensor = cml.model_.W.state_dict()["weight"]
    H_as_tensor = cml.model_.H.state_dict()["weight"]

    W_as_tensor_approximated = cml.approximate_W(s._validation_data_in.binary_values, W_as_tensor, H_as_tensor)

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(W_as_tensor.detach().cpu().numpy(), W_as_tensor_approximated.detach().cpu().numpy())


def test_covariance_loss():
    ct = CMLTorch(10000, 1000, 200)

    loss = covariance_loss(ct.H, ct.W).detach().numpy()

    # Embeddings are initialized to be zero mean, ct.std standard deviation.
    np.testing.assert_almost_equal(abs(loss), ct.std, decimal=1)


def test_cml_save_load(cml_save, larger_matrix):

    cml_save.fit(larger_matrix, (larger_matrix, larger_matrix))
    assert os.path.isfile(cml_save.filename)

    os.remove(cml_save.filename)


def test_cleanup(larger_matrix):
    def inner():
        a = CML(
            100,  # num_components
            1.9,  # margin
            0.1,  # learning_rate
            2,  # num_epochs
        )
        a._init_model(larger_matrix)
        assert os.path.isfile(a.best_model_.name)
        return a.best_model_.name

    n = inner()
    assert not os.path.isfile(n)
