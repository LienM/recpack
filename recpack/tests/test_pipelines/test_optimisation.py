import pytest

from hyperopt import hp

from recpack.pipelines import HyperoptInfo


def test_check_parameters_hyperopt():
    with pytest.raises(ValueError) as e:
        HyperoptInfo(space={"K": hp.uniformint("K", 50, 1000)})

    assert e.match("infinite")
