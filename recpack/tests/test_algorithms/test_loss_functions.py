# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

from recpack.algorithms.loss_functions import (
    warp_loss_wrapper,
    warp_loss,
    covariance_loss,
    bpr_loss,
    bpr_loss_wrapper,
    skipgram_negative_sampling_loss,
    bpr_max_loss,
    top1_loss,
    top1_max_loss,
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


def test_warp_loss_wrapper():

    X_true = csr_matrix([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

    X_pred = csr_matrix([[0.2, 0.2, 0], [0.8, 1, 0.9], [1, 1, 0.95]])

    margin = 0.1
    num_negatives = 2

    loss = warp_loss_wrapper(X_true, X_pred, batch_size=1, num_negatives=num_negatives, margin=margin, exact=True)

    expected_loss = (
        # First term, correctly classified (distance to pos < distance to neg)
        math.log((0 * 3 / num_negatives) + 1) * (0 - 0.2 + margin)
        # 2 items were wrongly classified
        + math.log((2 * 3 / num_negatives) + 1) * (1 - 0.8 + margin)
        # Wrong classification was within margin
        + math.log((2 * 3 / num_negatives) + 1) * (1 - 0.95 + margin)
    ) / 3  # Take the mean loss per item

    np.testing.assert_almost_equal(loss, expected_loss)


def test_covariance_loss():
    W_as_tensor = torch.FloatTensor([[0.5, 0.4], [0.1, 0.3]])
    H_as_tensor = torch.FloatTensor([[0.4, 0.4], [0.2, 0.9]])

    W = nn.Embedding.from_pretrained(W_as_tensor)
    H = nn.Embedding.from_pretrained(H_as_tensor)

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


def test_skipgram_negative_sampling_loss():
    pos_sim_as_tensor = torch.FloatTensor([0.6, 0.3])
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
    loss = skipgram_negative_sampling_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    neg_loss = np.log(sigmoid(-0.1)) + np.log(sigmoid(-0.3)) + np.log(sigmoid(-0.6))

    expected_loss = np.array([-(np.log(sigmoid(0.6)) + neg_loss), -(np.log(sigmoid(0.3)) + neg_loss)]).mean()

    np.testing.assert_almost_equal(loss, expected_loss, decimal=3)


# -------------------- BPR


def test_bpr_loss_2d_1dim():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]])
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]])

    loss = bpr_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    expected_loss = -(np.log(sigmoid(0.5)) + np.log(sigmoid(0)) + np.log(sigmoid(-0.5))) / 3
    np.testing.assert_almost_equal(loss, expected_loss)


def test_bpr_loss_1d():
    pos_sim_as_tensor = torch.FloatTensor([0.6, 0.3, 0.1])
    neg_sim_as_tensor = torch.FloatTensor([0.1, 0.3, 0.6])

    loss = bpr_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    expected_loss = -(np.log(sigmoid(0.5)) + np.log(sigmoid(0)) + np.log(sigmoid(-0.5))) / 3
    np.testing.assert_almost_equal(loss, expected_loss)


def test_bpr_loss_2d():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]).t()

    loss = bpr_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    expected_loss = -(np.log(sigmoid(0.5)) + np.log(sigmoid(0)) + np.log(sigmoid(-0.5))) / 3
    np.testing.assert_almost_equal(loss, expected_loss)


# -------------------- BPR MAX


def test_bpr_max_loss_2d_1dim():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]])
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]])

    reg = 2
    loss = bpr_max_loss(pos_sim_as_tensor, neg_sim_as_tensor, reg=reg)

    s = neg_sim_as_tensor.softmax(dim=0)

    # Mean over all "users", 1 negative sample for every positive.
    norm_penalty = (
        (s[0, 0] * (neg_sim_as_tensor[0, 0] ** 2))
        + (s[0, 1] * (neg_sim_as_tensor[0, 1] ** 2))
        + (s[0, 2] * (neg_sim_as_tensor[0, 2] ** 2))
    ) / 3

    expected_loss = (
        -(np.log(s[0, 0] * sigmoid(0.5)) + np.log(s[0, 1] * sigmoid(0)) + np.log(s[0, 1] * sigmoid(-0.5))) / 3
    ) + reg * norm_penalty
    np.testing.assert_almost_equal(loss, expected_loss)

    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]]).t()

    reg = 2
    loss = bpr_max_loss(pos_sim_as_tensor, neg_sim_as_tensor, reg=reg)

    s = neg_sim_as_tensor.softmax(dim=1)

    # Mean over all "users", 1 negative sample for every positive.
    norm_penalty = (
        (s[0, 0] * (neg_sim_as_tensor[0, 0] ** 2))
        + (s[1, 0] * (neg_sim_as_tensor[1, 0] ** 2))
        + (s[2, 0] * (neg_sim_as_tensor[2, 0] ** 2))
    ) / 3

    expected_loss = (
        -(np.log(s[0, 0] * sigmoid(0.5)) + np.log(s[1, 0] * sigmoid(0)) + np.log(s[2, 0] * sigmoid(-0.5))) / 3
    ) + reg * norm_penalty
    np.testing.assert_almost_equal(loss, expected_loss)


def test_bpr_max_loss_1d():
    pos_sim_as_tensor = torch.FloatTensor([0.6, 0.3, 0.1])
    neg_sim_as_tensor = torch.FloatTensor([0.1, 0.3, 0.6])

    reg = 2
    loss = bpr_max_loss(pos_sim_as_tensor, neg_sim_as_tensor, reg=reg)

    s = neg_sim_as_tensor.unsqueeze(-1).softmax(dim=1)

    # Mean over all "users", 1 negative sample for every positive.
    norm_penalty = (
        (s[0] * (neg_sim_as_tensor[0] ** 2))
        + (s[1] * (neg_sim_as_tensor[1] ** 2))
        + (s[2] * (neg_sim_as_tensor[2] ** 2))
    ) / 3

    expected_loss = (
        -(np.log(s[0] * sigmoid(0.5)) + np.log(s[1] * sigmoid(0)) + np.log(s[2] * sigmoid(-0.5))) / 3
    ) + reg * norm_penalty
    np.testing.assert_almost_equal(loss, expected_loss)


def test_bpr_max_loss_2d():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]).t()

    reg = 2
    loss = bpr_max_loss(pos_sim_as_tensor, neg_sim_as_tensor, reg=reg)

    s = neg_sim_as_tensor.softmax(dim=1)

    norm_penalty0 = (s[0, 0] * (neg_sim_as_tensor[0, 0] ** 2)) + ((s[0, 1] * (neg_sim_as_tensor[0, 1] ** 2)))

    norm_penalty1 = (s[1, 0] * (neg_sim_as_tensor[1, 0] ** 2)) + ((s[1, 1] * (neg_sim_as_tensor[1, 1] ** 2)))

    norm_penalty2 = (s[2, 0] * (neg_sim_as_tensor[2, 0] ** 2)) + ((s[2, 1] * (neg_sim_as_tensor[2, 1] ** 2)))

    # Take the mean across samples
    norm_penalty = reg * (norm_penalty0 + norm_penalty1 + norm_penalty2) / 3

    # Multiply by 2 because two negatives (with the same value)
    expected_loss = (
        -(np.log(s[0, 0] * sigmoid(0.5) * 2) + np.log(s[1, 0] * sigmoid(0) * 2) + np.log(s[2, 0] * sigmoid(-0.5) * 2))
    ) / 3 + norm_penalty

    np.testing.assert_almost_equal(loss, expected_loss)


# ------------------- TOP-1 MAX


def test_top1_max_loss_2d_1dim():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]])
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]])

    loss = top1_max_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    s = neg_sim_as_tensor.softmax(dim=0)

    expected_loss = (
        s[0, 0] * (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 0] ** 2))
        + s[0, 1] * (sigmoid(0) + sigmoid(neg_sim_as_tensor[0, 1] ** 2))
        + s[0, 2] * (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[0, 2] ** 2))
    ) / 3

    np.testing.assert_almost_equal(loss, expected_loss)

    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]]).t()

    loss = top1_max_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    s = neg_sim_as_tensor.softmax(dim=1)

    expected_loss = (
        s[0, 0] * (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 0] ** 2))
        + s[1, 0] * (sigmoid(0) + sigmoid(neg_sim_as_tensor[1, 0] ** 2))
        + s[2, 0] * (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[2, 0] ** 2))
    ) / 3

    np.testing.assert_almost_equal(loss, expected_loss)


def test_top1_max_loss_1d():
    pos_sim_as_tensor = torch.FloatTensor([0.6, 0.3, 0.1])
    neg_sim_as_tensor = torch.FloatTensor([0.1, 0.3, 0.6])

    loss = top1_max_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    s = neg_sim_as_tensor.unsqueeze(-1).softmax(dim=1)

    expected_loss = (
        s[0, 0] * sigmoid(-0.5)
        + sigmoid(neg_sim_as_tensor[0] ** 2)
        + s[1, 0] * sigmoid(0)
        + sigmoid(neg_sim_as_tensor[1] ** 2)
        + s[2, 0] * sigmoid(0.5)
        + sigmoid(neg_sim_as_tensor[2] ** 2)
    ) / 3

    np.testing.assert_almost_equal(loss, expected_loss)


def test_top1_max_loss_2d():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]).t()

    loss = top1_max_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    s = neg_sim_as_tensor.softmax(dim=1)

    loss_term1 = (s[0, 0] * (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 0] ** 2))) + (
        s[0, 1] * (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 1] ** 2))
    )

    loss_term2 = (s[1, 0] * (sigmoid(0) + sigmoid(neg_sim_as_tensor[1, 0] ** 2))) + (
        s[1, 1] * (sigmoid(0) + sigmoid(neg_sim_as_tensor[1, 1] ** 2))
    )

    loss_term3 = (s[2, 0] * (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[2, 0] ** 2))) + (
        s[2, 1] * (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[2, 1] ** 2))
    )

    expected_loss = (loss_term1 + loss_term2 + loss_term3) / 3

    np.testing.assert_almost_equal(loss, expected_loss)


# ------------------- TOP1


def test_top1_loss_2d_1dim():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]])
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]])

    loss = top1_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    expected_loss = (
        (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 0] ** 2))
        + (sigmoid(0) + sigmoid(neg_sim_as_tensor[0, 1] ** 2))
        + (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[0, 2] ** 2))
    ) / 3

    np.testing.assert_almost_equal(loss, expected_loss)

    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6]]).t()

    loss = top1_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    expected_loss = (
        (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 0] ** 2))
        + (sigmoid(0) + sigmoid(neg_sim_as_tensor[1, 0] ** 2))
        + (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[2, 0] ** 2))
    ) / 3

    np.testing.assert_almost_equal(loss, expected_loss)


def test_top1_loss_1d():
    pos_sim_as_tensor = torch.FloatTensor([0.6, 0.3, 0.1])
    neg_sim_as_tensor = torch.FloatTensor([0.1, 0.3, 0.6])

    loss = top1_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    expected_loss = (
        sigmoid(-0.5)
        + sigmoid(neg_sim_as_tensor[0] ** 2)
        + sigmoid(0)
        + sigmoid(neg_sim_as_tensor[1] ** 2)
        + sigmoid(0.5)
        + sigmoid(neg_sim_as_tensor[2] ** 2)
    ) / 3

    np.testing.assert_almost_equal(loss, expected_loss)


def test_top1_loss_2d():
    pos_sim_as_tensor = torch.FloatTensor([[0.6, 0.3, 0.1]]).t()
    neg_sim_as_tensor = torch.FloatTensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]).t()

    loss = top1_loss(pos_sim_as_tensor, neg_sim_as_tensor)

    s = neg_sim_as_tensor.softmax(dim=1)

    expected_loss = (
        # Negative sample 1, user 1
        (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 0] ** 2))
        # Negative sample 2, user 2
        + (sigmoid(-0.5) + sigmoid(neg_sim_as_tensor[0, 1] ** 2))
        + (sigmoid(0) + sigmoid(neg_sim_as_tensor[1, 0] ** 2))
        + (sigmoid(0) + sigmoid(neg_sim_as_tensor[1, 1] ** 2))
        + (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[2, 0] ** 2))
        + (sigmoid(0.5) + sigmoid(neg_sim_as_tensor[2, 1] ** 2))
    ) / 6

    np.testing.assert_almost_equal(loss, expected_loss)


def test_bpr_loss_wrapper(X_true, X_pred):
    # Tuples are user, pos_item, neg_item
    xij_samples = np.array([[0, 1, 0], [0, 2, 0], [1, 0, 2], [1, 1, 2]])

    x = xij_samples[:, 0]
    i = xij_samples[:, 1]
    j = xij_samples[:, 2]

    pos_samples = X_pred[x, i]
    neg_samples = X_pred[x, j]

    expected_loss = bpr_loss(torch.FloatTensor(pos_samples), torch.FloatTensor(neg_samples))

    # Using more samples than positives this will make the estimate more stable,
    # And assert below should match.
    loss = bpr_loss_wrapper(X_true, X_pred, batch_size=1, sample_size=1000, exact=True)

    np.testing.assert_almost_equal(loss, expected_loss, decimal=2)
