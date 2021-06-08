import numpy as np
import pandas as pd
import pytest

from scipy.sparse import csr_matrix
from recpack.data.matrix import InteractionMatrix
from recpack.algorithms.gru4rec import GRU4Rec
from recpack.tests.test_algorithms.util import assert_changed, assert_same


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def matrix_sessions() -> InteractionMatrix:
    # (user, time) matrix, non-zero entries are item ids
    user_time = csr_matrix(
        [
            # 0  1  2  3  4  5  6  7
            [0, 1, 2, 0, 0, 0, 0, 0],
            [1, 2, 0, 1, 3, 1, 0, 0],
            [1, 2, 1, 2, 1, 0, 2, 1],
            [1, 3, 1, 2, 1, 0, 2, 1],
            [1, 2, 1, 2, 1, 2, 1, 2],
        ]
    )
    user_ids, timestamps = user_time.nonzero()
    item_ids = user_time.data
    df = pd.DataFrame(
        {
            USER_IX: user_ids,
            ITEM_IX: item_ids,
            TIMESTAMP_IX: timestamps,
        }
    )
    return InteractionMatrix(
        df, user_ix=USER_IX, item_ix=ITEM_IX, timestamp_ix=TIMESTAMP_IX
    )


@pytest.fixture(scope="function")
def session_rnn():
    rnn = GRU4Rec(seed=42, batch_size=1, embedding_size=5, hidden_size=10)
    return rnn


def test_session_rnn_training_epoch(session_rnn, matrix_sessions):
    device = session_rnn.device
    session_rnn._init_model(matrix_sessions)


    # Each training epoch should update the parameters
    for _ in range(5):
        params = [
            np for np in session_rnn.model_.named_parameters() if np[1].requires_grad
        ]
        params_before = [(name, p.clone()) for (name, p) in params]

        session_rnn._train_epoch(matrix_sessions)
        assert_changed(params_before, params, device)


# TODO Test if we update with information for all users when bptt > 1.  


def test_session_rnn_evaluation_epoch(session_rnn, matrix_sessions):
    device = session_rnn.device

    session_rnn.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    # Model evaluation should have no effect on parameters
    for _ in range(5):
        params = [
            np for np in session_rnn.model_.named_parameters() if np[1].requires_grad
        ]
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
