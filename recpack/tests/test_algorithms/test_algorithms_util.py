import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor

from recpack.algorithms.util import (
    naive_sparse2tensor,
    naive_tensor2sparse,
    sample_rows,
    get_batches
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
        print(batch)


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
