import numpy as np
import pandas as pd
import pytest
import torch
from torch.autograd import Variable
import torch.nn as nn
from scipy.sparse import csr_matrix


from recpack.matrix import InteractionMatrix

INPUT_SIZE = 1000
USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def input_size():
    return INPUT_SIZE


@pytest.fixture(scope="function")
def inputs():
    torch.manual_seed(400)
    return Variable(torch.randn(INPUT_SIZE, INPUT_SIZE))


@pytest.fixture(scope="function")
def targets():
    torch.manual_seed(400)
    return Variable(torch.randint(0, 2, (INPUT_SIZE,))).long()


@pytest.fixture(scope="function")
def small_mat_unigram():
    data = {
        TIMESTAMP_IX: np.random.randint(0, 10, size=10),
        ITEM_IX: [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
        USER_IX: np.random.randint(0, 10, size=10),
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture(scope="function")
def pageviews():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [1, 2, 1, 1, 1, 1],
    )

    pv = csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


@pytest.fixture(scope="function")
def pageviews_for_pairwise():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 1, 1, 1, 3, 3, 4, 4],
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    pv = csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


@pytest.fixture(scope="function")
def pageviews_interaction_m(pageviews):
    return InteractionMatrix.from_csr_matrix(pageviews)


@pytest.fixture(scope="function")
def purchases():
    pur_users, pur_items, pur_values = (
        [0, 2],
        [0, 4],
        [1, 1],
    )

    pur = csr_matrix((pur_values, (pur_users, pur_items)), shape=(10, 5))

    return pur


@pytest.fixture(scope="function")
def larger_matrix():
    num_interactions = 2000
    num_users = 500
    num_items = 500

    np.random.seed(400)

    pv_users, pv_items, pv_values = (
        [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        [1] * num_interactions,
    )

    pv = csr_matrix((pv_values, (pv_users, pv_items)), shape=(num_users + 200, num_items))

    return pv


@pytest.fixture(scope="function")
def data():
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )

    pred = csr_matrix((pred_values, (pred_users, pred_items)), shape=(10, 5))

    return pred


@pytest.fixture(scope="function")
def p2v_embedding():
    # the target values for our predictions
    # we have five users, the target is the last item the user bought
    # values = [1] * 5
    # users = [0, 1, 2, 3, 4]
    # items = [0, 1, 2, 3, 4]
    # target = sp.csr_matrix((values, (users, items)))
    # target = InteractionMatrix.from_csr_matrix(target)

    # pre-defined embedding vectors
    embedding = [
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.4, 0.4, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    embedding = np.array(embedding)
    embedding = torch.from_numpy(embedding)
    embedding = nn.Embedding.from_pretrained(embedding)
    return embedding
