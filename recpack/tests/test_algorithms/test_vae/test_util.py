from recpack.algorithms.vae.util import (
    StoppingCriterion,
    naive_sparse2tensor,
    naive_tensor2sparse
)
from recpack.metrics import NDCGK

from scipy.sparse import csr_matrix
from torch import Tensor
import pytest


def test_stopping_criterion(larger_matrix):
    crit = StoppingCriterion(NDCGK, 100)

    crit.calculate(larger_matrix, larger_matrix)
    # We don't really care about the resulting scores

    assert crit.value != 0

    v = crit.value

    # First time computing :D
    assert crit.is_best

    crit.reset()

    with pytest.raises(AttributeError):
        crit.value
    assert crit.best_value == v

    crit.calculate(larger_matrix, larger_matrix)

    assert crit.value != 0
    assert crit.value == v  #  Should be a deterministic metric


def test_csr_tensor_conversions(larger_matrix):
    assert isinstance(larger_matrix, csr_matrix)
    tensor = naive_sparse2tensor(larger_matrix)
    assert isinstance(tensor, Tensor)

    csr_again = naive_tensor2sparse(tensor)

    assert isinstance(csr_again, csr_matrix)
