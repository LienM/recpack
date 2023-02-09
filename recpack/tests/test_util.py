# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import scipy.sparse

from recpack.util import get_top_K_ranks, get_top_K_values


def test_get_topK_ranks():
    state = np.random.RandomState(13940)

    mat = scipy.sparse.random(2, 100, density=0.10, random_state=state).tocsr()

    top_k_ranks = get_top_K_ranks(mat, 20)

    max_ix_row_0 = np.argmax(mat[0, :])
    assert top_k_ranks[0, max_ix_row_0] == 1

    max_ix_row_1 = np.argmax(mat[1, :])
    assert top_k_ranks[1, max_ix_row_1] == 1


def test_get_all_topK_ranks(data, ranked_data_complete):
    top_k_ranks = get_top_K_ranks(data, None)
    np.testing.assert_almost_equal(
        top_k_ranks.todense(), ranked_data_complete.todense()
    )


def test_get_topK_ranks_no_reco():
    state = np.random.RandomState(13940)

    mat = scipy.sparse.random(2, 100, density=0.10, random_state=state).tocsr()

    top_k_ranks = get_top_K_ranks(mat, 20)

    max_ix_row_0 = np.argmax(mat[0, :])
    assert top_k_ranks[0, max_ix_row_0] == 1

    max_ix_row_1 = np.argmax(mat[1, :])
    assert top_k_ranks[1, max_ix_row_1] == 1


def test_get_top_K_values(data_knn):
    top_k_values = get_top_K_values(data_knn, 2)

    topK_users, topK_items, topK_values = (
        [0, 0, 2, 2],
        [0, 2, 3, 4],
        [0.3, 0.2, 0.3, 0.5],
    )

    topK = scipy.sparse.csr_matrix(
        (topK_values, (topK_users, topK_items)), shape=(10, 5)
    )

    np.testing.assert_almost_equal(topK.todense(), top_k_values.todense())
