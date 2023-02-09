# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

from hyperopt import hp

from recpack.pipelines import HyperoptInfo


def test_check_parameters_hyperopt():
    with pytest.raises(ValueError) as e:
        HyperoptInfo(space={"K": hp.uniformint("K", 50, 1000)})

    assert e.match("infinite")
