# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pytest

from scipy.sparse import csr_matrix
import torch

from recpack.algorithms.gru4rec import GRU4RecNegSampling
from recpack.tests.test_algorithms.util import assert_changed, assert_same
from recpack.tests.test_algorithms.test_loss_functions import sigmoid


@pytest.fixture(scope="function")
def session_rnn():
    rnn = GRU4RecNegSampling(
        seed=42,
        batch_size=3,
        num_components=5,
        hidden_size=10,
        num_negatives=25,
        bptt=2,
        learning_rate=0.1,
        loss_fn="bpr",
        keep_last=True,
    )
    return rnn


@pytest.fixture(scope="function")
def session_rnn_topK():
    rnn = GRU4RecNegSampling(
        seed=42,
        batch_size=3,
        num_components=5,
        hidden_size=10,
        num_negatives=25,
        bptt=2,
        learning_rate=0.1,
        loss_fn="bpr",
        keep_last=True,
        predict_topK=1,
    )
    return rnn


def test_session_rnn_compute_loss(session_rnn):

    output = torch.FloatTensor(
        [
            [[0, 0.1, 0.8, 0.1, 0], [0, 0.8, 0.2, 0, 0]],
            [[0, 0.1, 0.8, 0.1, 0], [0, 0.8, 0.2, 0, 0]],
            [[0, 0.1, 0.8, 0.1, 0], [0, 0, 0, 0, 1]],
        ]
    )

    targets_chunk = torch.LongTensor([[2, 1], [2, 1], [2, 4]])

    negatives_chunk = torch.LongTensor([[[1, 3], [2, 2]], [[1, 3], [2, 3]], [[3, 0], [3, 3]]])

    true_input_mask = torch.BoolTensor([[True, True], [True, True], [True, True]])

    loss = session_rnn._compute_loss(output, targets_chunk, negatives_chunk, true_input_mask)

    expected_loss = (
        -(
            np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.2))
            + np.log(sigmoid(0.8 - 0.2))
            + np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.2))
            + np.log(sigmoid(0.8 - 0.0))
            + np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.0))
            + np.log(sigmoid(1.0 - 0.0))
            + np.log(sigmoid(1.0 - 0.0))
        )
        / 12
    )
    np.testing.assert_almost_equal(loss, expected_loss)

    # Block out the middle element
    true_input_mask = torch.BoolTensor([[True, True], [False, False], [True, True]])
    loss = session_rnn._compute_loss(output, targets_chunk, negatives_chunk, true_input_mask)

    expected_loss = (
        -(
            np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.2))
            + np.log(sigmoid(0.8 - 0.2))
            +
            # np.log(sigmoid(0.8 - 0.1)) + np.log(sigmoid(0.8 - 0.1)) +
            # np.log(sigmoid(0.8 - 0.2)) + np.log(sigmoid(0.8 - 0.0)) +
            np.log(sigmoid(0.8 - 0.1))
            + np.log(sigmoid(0.8 - 0.0))
            + np.log(sigmoid(1.0 - 0.0))
            + np.log(sigmoid(1.0 - 0.0))
        )
        / 8
    )
    np.testing.assert_almost_equal(loss, expected_loss)


def test_session_rnn_training_epoch(session_rnn, matrix_sessions):
    device = session_rnn.device
    session_rnn._init_model(matrix_sessions)

    # Each training epoch should update the parameters
    for _ in range(5):
        params = [np for np in session_rnn.model_.named_parameters() if np[1].requires_grad]
        params_before = [(name, p.clone()) for (name, p) in params]

        session_rnn._train_epoch(matrix_sessions)
        assert_changed(params_before, params, device)


def test_session_rnn_evaluation_epoch(session_rnn, matrix_sessions):
    device = session_rnn.device

    session_rnn.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    # Model evaluation should have no effect on parameters
    for _ in range(5):
        params = [np for np in session_rnn.model_.named_parameters() if np[1].requires_grad]
        params_before = [(name, p.clone()) for (name, p) in params]

        session_rnn._evaluate(matrix_sessions, matrix_sessions)
        assert_same(params_before, params, device)


def test_session_rnn_predict(session_rnn, matrix_sessions):
    session_rnn.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    X_pred = session_rnn.predict(matrix_sessions)
    scores = X_pred.toarray()

    top_item = scores.argmax(axis=1)

    # Prediction matrix should have same shape as input matrix
    assert isinstance(X_pred, csr_matrix)
    assert X_pred.shape == matrix_sessions.shape

    # All users with a history should have predictions
    assert set(matrix_sessions.values.nonzero()[0]) == set(X_pred.nonzero()[0])

    # All items should have a score
    assert len(set(X_pred.nonzero()[1])) == matrix_sessions.shape[1]

    # Rnn should be able to learn simple repeating patterns
    assert top_item[0] == 1
    assert top_item[1] == 2
    assert top_item[2] == 2
    assert top_item[3] == 2
    assert top_item[4] == 1


def test_session_rnn_predict_topK(session_rnn_topK, matrix_sessions):
    session_rnn_topK.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    X_pred = session_rnn_topK.predict(matrix_sessions)
    scores = X_pred.toarray()

    top_item = scores.argmax(axis=1)

    # Prediction matrix should have same shape as input matrix
    assert isinstance(X_pred, csr_matrix)
    assert X_pred.shape == matrix_sessions.shape

    # All users with a history should have predictions
    assert set(matrix_sessions.values.nonzero()[0]) == set(X_pred.nonzero()[0])

    # Each user should receive only a single recommendation
    assert X_pred.nonzero()[1].shape[0] == len(set(matrix_sessions.nonzero()[0]))

    # Rnn should be able to learn simple repeating patterns
    assert top_item[0] == 1
    assert top_item[1] == 2
    assert top_item[2] == 2
    assert top_item[3] == 2
    assert top_item[4] == 1


def test_fit_no_interaction_matrix(session_rnn_topK, mat):
    with pytest.raises(TypeError):
        session_rnn_topK.fit(mat.binary_values, (mat, mat))
    with pytest.raises(TypeError):
        session_rnn_topK.fit(mat, (mat.binary_values, mat))
    with pytest.raises(TypeError):
        session_rnn_topK.fit(mat, (mat, mat.binary_values))


def test_fit_no_timestamps(session_rnn_topK, mat):
    with pytest.raises(ValueError):
        session_rnn_topK.fit(mat.eliminate_timestamps(), (mat, mat))
    with pytest.raises(ValueError):
        session_rnn_topK.fit(mat, (mat.eliminate_timestamps(), mat))
    with pytest.raises(ValueError):
        session_rnn_topK.fit(mat, (mat, mat.eliminate_timestamps()))


def test_predict_no_interaction_matrix(session_rnn_topK, mat):
    session_rnn_topK.fit(mat, (mat, mat))
    with pytest.raises(TypeError):
        session_rnn_topK.predict(mat.binary_values)


def test_predict_no_timestamps(session_rnn_topK, mat):
    session_rnn_topK.fit(mat, (mat, mat))
    with pytest.raises(ValueError):
        session_rnn_topK.predict(mat.eliminate_timestamps())
