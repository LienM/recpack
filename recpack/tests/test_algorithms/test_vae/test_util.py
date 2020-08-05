from recpack.algorithms.vae.util import StoppingCriterion

def test_stopping_criterion(larger_matrix):
    crit = StoppingCriterion(100)

    crit.calculate(larger_matrix, larger_matrix)
    # We don't really care about the resulting scores

    assert crit.value != 0
    assert crit.num_users != 0

    v = crit.value

    # First time computing :D
    assert crit.is_best

    crit.reset()
    assert crit.value == 0
    assert crit.num_users == 0
    assert crit.best_value == v
