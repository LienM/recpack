# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn


from recpack.matrix import InteractionMatrix


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def X_in():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 1],
        [1, 2, 1, 1, 1, 1],
    )

    pv = csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


@pytest.fixture(scope="function")
def X_in_interaction_m(X_in):
    return InteractionMatrix.from_csr_matrix(X_in)


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
