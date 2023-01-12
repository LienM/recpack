# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from typing import Dict, Any, List

from sklearn.model_selection import ParameterGrid


class OptimisationInfo:
    """Base class for Optimisation Info."""
    pass


class GridSearchInfo(OptimisationInfo):
    """Info for a grid search optimisation.

    :param params: A dict which for each key contains a list of values to try.
    :type params: Dict[str, List[Any]]
    """

    def __init__(self, params: Dict[str, List[Any]]):
        self._grid = params

    @property
    def grid(self) -> ParameterGrid:
        """Sklearn ParameterGrid which will be used for optimisation."""
        return ParameterGrid(self._grid)


class HyperoptInfo(OptimisationInfo):
    """Information for hyperopt parameter optimisation

    Either specify a minimum number of evaluations using ``max_evals`` or specify a
    max amount of seconds to run optimisation for ``timeout``.
    If both are specified, the first condition met will cause the optimisation to stop.

    For a detailed tutorial on using hyperopt see :ref:`guides-hyperopt`

    :param space: hyperopt parameter space, should be a dict from hyper_parameter to the hp definition of its value
        (eg. hp.uniformint)
    :type space: Dict[str, Any]
    :param timeout: Number of seconds to optimise before stopping, defaults to None
    :type timeout: int, optional
    :param max_evals: Maximum number of parameter combinations to try, defaults to None
    :type max_evals: int, optional
    :raises ValueError: When both timeout or max_evals are None, avoiding an infinite optimisation loop.
    """

    def __init__(self, space: Dict[str, Any], timeout: int = None, max_evals: int = None):
        self.space = space
        self.timeout = timeout
        self.max_evals = max_evals

        if self.timeout is None and self.max_evals is None:
            raise ValueError(
                "Please specify max_evals and/or timeout, otherwise optimisation will enter an infinite loop."
            )
    
