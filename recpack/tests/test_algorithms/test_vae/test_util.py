from recpack.algorithms.vae.util import StoppingCriterion
from recpack.metrics import NDCGK
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
    assert crit.value == v # Should be a deterministic metric
