# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor

from recpack.algorithms.util import (
    csr_from_rows,
    get_batches,
    invert,
    naive_sparse2tensor,
    naive_tensor2sparse,
    sample_rows,
    get_batches,
    union_csr_matrices,
)


def test_get_batches():

    test_array = range(0, 500)

    cnt = 0

    for batch in get_batches(test_array, batch_size=10):
        assert (len(batch) == 10 or len(batch) < 10) and len(batch) != 0
        cnt += len(batch)

    assert cnt == 500

    cnt = 0

    for batch in get_batches(iter(test_array), batch_size=10):
        assert (len(batch) == 10 or len(batch) < 10) and len(batch) != 0
        cnt += len(batch)

    assert cnt == 500

    it = get_batches(iter(test_array), batch_size=10)

    assert next(it) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert next(it) == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_csr_tensor_conversions(larger_matrix):
    assert isinstance(larger_matrix, csr_matrix)
    tensor = naive_sparse2tensor(larger_matrix)
    assert isinstance(tensor, Tensor)

    csr_again = naive_tensor2sparse(tensor)

    assert isinstance(csr_again, csr_matrix)


def test_sample_rows():

    users = [0, 0, 1, 1, 2]
    items = [1, 2, 2, 3, 4]
    values = [1 for i in items]
    mat_1 = csr_matrix((values, (users, items)))

    # Different values, makes assertions more correct
    mat_2 = csr_matrix(([v / 2 for v in values], (users, items)))

    s_1, s_2 = sample_rows(mat_1, mat_2, sample_size=2)
    np.testing.assert_array_equal(s_1.nonzero(), s_2.nonzero())
    assert len(set(s_1.nonzero()[0])) == 2
    np.testing.assert_array_equal(s_1[s_1.nonzero()], 1)
    np.testing.assert_array_almost_equal(s_2[s_2.nonzero()], 0.5)


def test_sample_rows_im(mat):
    (sample,) = sample_rows(mat, sample_size=2)
    assert sample.num_active_users == 2


def test_sample_rows_im_2(larger_mat):
    N_RUNS = 10
    N_ROWS = 50
    for i in range(N_RUNS):
        (sample,) = sample_rows(larger_mat, sample_size=N_ROWS)

        assert sample.num_active_users == N_ROWS


def test_get_batches_small():
    users = list(range(100))
    batch_size = 12

    total = 0
    for batch in get_batches(users, batch_size=batch_size):
        total += len(batch)

    assert total == len(users)


def test_union_csr_matrices():
    # fmt:off
    a = csr_matrix(np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 0]
    ]))

    b = csr_matrix(np.array([
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]))

    expected = csr_matrix(np.array([
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ]))

    # fmt:on
    combined = union_csr_matrices(a, b)

    np.testing.assert_equal(combined.toarray(), expected.toarray())


def test_invert():
    a = np.array([1, 2, 3, 0])

    expected = np.array([1, 1 / 2, 1 / 3, 0])

    inv = invert(a)

    np.testing.assert_almost_equal(inv, expected)


def test_csr_from_rows():
    rows = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]]))

    result = csr_from_rows(rows, [4, 1], (6, 3))

    expected = np.zeros((6, 3))
    expected[4] = [1, 0, 2]
    expected[1] = [0, 3, 0]

    assert isinstance(result, csr_matrix)
    np.testing.assert_array_equal(result.toarray(), expected)


def test_csr_from_rows_sorted_indices():
    rows = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]]))

    result = csr_from_rows(rows, [1, 4], (6, 3))

    expected = np.zeros((6, 3))
    expected[1] = [1, 0, 2]
    expected[4] = [0, 3, 0]

    np.testing.assert_array_equal(result.toarray(), expected)


def test_csr_from_rows_matches_lil_construction():
    # The utility replaces the lil_matrix allocate-and-assign pattern:
    # results must be identical for arbitrary inputs.
    from scipy.sparse import lil_matrix, random as sparse_random

    rng = np.random.default_rng(42)
    for _ in range(5):
        num_rows, num_cols = 50, 20
        selected = rng.choice(num_rows, size=10, replace=False)

        rows = csr_matrix(sparse_random(len(selected), num_cols, density=0.3, random_state=rng))

        reference = lil_matrix((num_rows, num_cols))
        reference[selected] = rows
        reference = reference.tocsr()

        result = csr_from_rows(rows, selected, (num_rows, num_cols))

        np.testing.assert_array_almost_equal(result.toarray(), reference.toarray())


def test_csr_from_rows_empty_rows():
    # Rows without any values should be handled correctly.
    rows = csr_matrix(np.array([[0, 0, 0], [1, 0, 0]]))

    result = csr_from_rows(rows, [2, 0], (4, 3))

    expected = np.zeros((4, 3))
    expected[0] = [1, 0, 0]

    np.testing.assert_array_equal(result.toarray(), expected)
