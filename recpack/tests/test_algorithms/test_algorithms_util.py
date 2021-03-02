from scipy.sparse import csr_matrix
from torch import Tensor

from recpack.algorithms.util import (
    naive_sparse2tensor,
    naive_tensor2sparse,
)


def test_csr_tensor_conversions(larger_matrix):
    assert isinstance(larger_matrix, csr_matrix)
    tensor = naive_sparse2tensor(larger_matrix)
    assert isinstance(tensor, Tensor)

    csr_again = naive_tensor2sparse(tensor)

    assert isinstance(csr_again, csr_matrix)
