import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

from recpack.algorithms.loss_functions import (
    warp_loss_metric,
    warp_loss,
    covariance_loss,
    bpr_loss,
    bpr_loss_metric,
)


@pytest.fixture(scope="function")
def X_true():
    users = [0, 0, 1, 1]
    items = [1, 2, 0, 1]
    values = [1 for i in items]
    return csr_matrix((values, (users, items)), shape=(2, 3))


@pytest.fixture(scope="function")
def X_pred():
    users = [0, 0, 0, 1, 1, 1]
    items = [0, 1, 2, 0, 1, 2]
    scores = [0, 0.3, 0.5, 0.2, 0.4, 0.5]
    return csr_matrix((scores, (users, items)), shape=(2, 3))


def test_warp_loss_metric():

    X_true = csr_matrix([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

    X_pred = csr_matrix([[0.2, 0.2, 0], [0.8, 1, 0.9], [1, 1, 0.95]])

    margin = 0.1
    U = 2

    loss = warp_loss_metric(
        X_true, X_pred, batch_size=1, U=U, margin=margin, exact=True
    )

    expected_loss = (
        # First term, correctly classified (distance to pos < distance to neg)
        math.log((0 * 3 / U) + 1) * (0 - 0.2 + margin)
        # 2 items were wrongly classified
        + math.log((2 * 3 / U) + 1) * (1 - 0.8 + margin)
        # Wrong classification was within margin
        + math.log((2 * 3 / U) + 1) * (1 - 0.95 + margin)
    ) / 3  # Take the mean loss per item

    np.testing.assert_almost_equal(loss, expected_loss)


def test_covariance_loss():
    W_as_tensor = torch.FloatTensor([[0.5, 0.4], [0.1, 0.3]])
    H_as_tensor = torch.FloatTensor([[0.4, 0.4], [0.2, 0.9]])

    W = nn.Embedding.from_pretrained(W_as_tensor)
    H = nn.Embedding.from_pretrained(H_as_tensor)

    print(next(W.parameters()))

    print(W_as_tensor.mean(dim=0))

    cov_loss = covariance_loss(W, H)

    # Different computation of loss:
    # Manually computed covariances:
    cov = np.array(
        [
            [0, -0.04 + 0.02, 0.02 + 0.01, -0.02 - 0.04],
            [-0.04 + 0.02, 0, -0.02 + 0.02, 0.02 - 0.08],
            [0.02 + 0.01, -0.02 + 0.02, 0, -0.01 - 0.04],
            [-0.02 - 0.04, 0.02 - 0.08, -0.01 - 0.04, 0],
        ]
    )
    # Checking that no error was made typing over the manual comp :smile:
    np.testing.assert_array_almost_equal(np.diag(cov), 0)
    np.testing.assert_array_equal(cov.T, cov)

    # expected value = sum(covariances) / ((num_users + num_items) * num_dimensions)
    expected_value = cov.sum() / ((2 + 2) * 2)

    np.testing.assert_almost_equal(cov_loss, expected_value)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def test_bpr_loss():
    pos_sim_as_tensor = torch.FloatTensor([0.6, 0.3, 0.1])
    neg_sim_as_tensor = torch.FloatTensor([0.1, 0.3, 0.6])

    loss = bpr_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    # x_uij[1] = 0.5
    # x_uij[2] = 0
    # x_uij[3] = -0.5

    expected_loss = (
        -(np.log(sigmoid(0.5)) + np.log(sigmoid(0)) + np.log(sigmoid(-0.5))) / 3
    )
    np.testing.assert_almost_equal(loss, expected_loss)


def test_bpr_loss_metric(X_true, X_pred):
    # Tuples are user, pos_item, neg_item
    xij_samples = np.array([[0, 1, 0], [0, 2, 0], [1, 0, 2], [1, 1, 2]])

    x = xij_samples[:, 0]
    i = xij_samples[:, 1]
    j = xij_samples[:, 2]

    pos_samples = X_pred[x, i]
    neg_samples = X_pred[x, j]

    # print(pos_samples, neg_samples)
    expected_loss = bpr_loss(
        torch.FloatTensor(pos_samples), torch.FloatTensor(neg_samples)
    )

    # Using more samples than positives this will make the estimate more stable,
    # And assert below should match.
    loss = bpr_loss_metric(X_true, X_pred, batch_size=1, sample_size=1000, exact=True)

    np.testing.assert_almost_equal(loss, expected_loss, decimal=2)
