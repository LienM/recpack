import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor

from recpack.algorithms.util import (
    naive_sparse2tensor,
    naive_tensor2sparse,
    normalize,
    sample,
)


def test_csr_tensor_conversions(larger_matrix):
    assert isinstance(larger_matrix, csr_matrix)
    tensor = naive_sparse2tensor(larger_matrix)
    assert isinstance(tensor, Tensor)

    csr_again = naive_tensor2sparse(tensor)

    assert isinstance(csr_again, csr_matrix)


def test_normalize():
    mat = csr_matrix(np.array([[1, 4, 5], [2, 3, 3]]))
    n_mat = normalize(mat)

    expected_values = np.array([[0.1, 0.4, 0.5], [2 / 8, 3 / 8, 3 / 8]])
    np.testing.assert_array_almost_equal(n_mat.toarray(), expected_values)


def test_sample():

    users = [0, 0, 1, 1, 2]
    items = [1, 2, 2, 3, 4]
    values = [1 for i in items]
    mat_1 = csr_matrix((values, (users, items)))

    # Different values, makes assertions more correct
    mat_2 = csr_matrix(([v / 2 for v in values], (users, items)))

    s_1, s_2 = sample(mat_1, mat_2, sample_size=2)
    np.testing.assert_array_almost_equal(s_1.nonzero(), s_2.nonzero())
    assert len(set(s_1.nonzero()[0])) == 2
    np.testing.assert_array_almost_equal(s_1[s_1.nonzero()], 1)
    np.testing.assert_array_almost_equal(s_2[s_2.nonzero()], 0.5)
