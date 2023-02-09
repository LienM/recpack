# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""

The scenarios module contains many of the most commonly encountered evaluation
scenarios in recommendation.

A scenario consists of a training and test dataset, and sometimes
also a validation dataset.
Both validation and test dataset are made up of two components:
a fold-in set of interactions that is used to predict another held-out
set of interactions.

Each scenario describes a complex situation, e.g. "Train on all user
interactions before time T, predict interactions after T+10 using interactions
from T+5 until T+10".

.. currentmodule:: recpack.scenarios

.. autosummary::
    :toctree: generated/

    Scenario

    Timed
    LastItemPrediction
    TimedLastItemPrediction
    WeakGeneralization
    StrongGeneralization
    StrongGeneralizationTimed
    StrongGeneralizationTimedMostRecent

A scenario is stateful. At initialization the parameters for the scenario are passed.
Only after calling :attr:`Scenario.split` given a :class:`recpack.matrix.InteractionMatrix`,
can splits be retrieved under
:attr:`Scenario.full_training_data`, :attr:`Scenario.validation_training_data`,
:attr:`Scenario.validation_data` and :attr:`Scenario.test_data`.


Splitters
------------

Splitters are the building blocks for the scenarios. A splitter performs a simple split into two InteractionMatrices
according to one, simple criterion, e.g. "Fold in is all interactions before T,
hold out all interactions after T".

Should you want to implement a new scenario that is not yet supported,
these splitters facilitate easy implementation.

.. currentmodule:: recpack.scenarios.splitters

.. autosummary::
    :toctree: generated/

    UserSplitter
    FractionInteractionSplitter
    TimestampSplitter
    StrongGeneralizationSplitter
    UserInteractionTimeSplitter
    MostRecentSplitter

"""

from recpack.scenarios.scenario_base import Scenario

from recpack.scenarios.last_item_prediction import LastItemPrediction
from recpack.scenarios.strong_generalization_timed_most_recent import StrongGeneralizationTimedMostRecent
from recpack.scenarios.strong_generalization_timed import StrongGeneralizationTimed
from recpack.scenarios.strong_generalization import StrongGeneralization
from recpack.scenarios.timed import Timed
from recpack.scenarios.timed_last_item_prediction import TimedLastItemPrediction
from recpack.scenarios.weak_generalization import WeakGeneralization
