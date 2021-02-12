import numpy as np
import pandas as pd
import pytest

from scipy.sparse import csr_matrix
from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, TIMESTAMP_IX, VALUE_IX
from recpack.algorithms.rnn.session_rnn import SessionRNN
from recpack.tests.test_algorithms.util import assert_changed, assert_same


@pytest.fixture(scope="function")
def data_m_sessions() -> DataM:
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
    return DataM(df)


@pytest.fixture(scope="function")
def session_rnn():
    rnn = SessionRNN(seed=42, batch_size=1, embedding_size=5, hidden_size=10)
    return rnn


def test_session_rnn_training_epoch(session_rnn, data_m_sessions):
    device = session_rnn.device

    session_rnn._init_random_state()
    session_rnn._init_model(data_m_sessions)
    session_rnn._init_training(data_m_sessions)

    # Each training epoch should update the parameters
    for _ in range(5):
        params = [np for np in session_rnn.model_.named_parameters() if np[1].requires_grad]
        params_before = [(name, p.clone()) for (name, p) in params]

        session_rnn._train_epoch(data_m_sessions)
        assert_changed(params_before, params, device)


def test_session_rnn_evaluation_epoch(session_rnn, data_m_sessions):
    device = session_rnn.device

    session_rnn.fit(data_m_sessions)

    # Model evaluation should have no effect on parameters
    for _ in range(5):
        params = [np for np in session_rnn.model_.named_parameters() if np[1].requires_grad]
        params_before = [(name, p.clone()) for (name, p) in params]

        session_rnn._evaluate((data_m_sessions, data_m_sessions))
        assert_same(params_before, params, device)


def test_session_rnn_predict(session_rnn, data_m_sessions):
    session_rnn.fit(data_m_sessions)

    X_pred = session_rnn.predict(data_m_sessions)
    scores = X_pred.toarray()
    top_item = scores.argmax(axis=1)

    # Prediction matrix should have same shape as input matrix
    assert isinstance(X_pred, csr_matrix)
    assert X_pred.shape == data_m_sessions.shape

    # All users with a history should have predictions
    assert set(data_m_sessions.values.nonzero()[0]) == set(X_pred.nonzero()[0])

    # All items should have a score
    assert len(set(X_pred.nonzero()[1])) == data_m_sessions.shape[1]

    # Rnn should be able to learn simple repeating patterns
    assert top_item[0] == 1
    assert top_item[1] == 2
    assert top_item[2] == 2
    assert top_item[3] == 2
    assert top_item[4] == 1
