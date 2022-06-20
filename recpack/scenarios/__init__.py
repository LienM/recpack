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
A splitter on the other hand performs a simple split into two InteractionMatrices
according to one, simple criterion, e.g. "Fold in is all interactions before T,
hold out all interactions after T".

Should you want to implement a new scenario that is not yet supported,
you can use the splitter submodule to help you build it from the ground up.

.. currentmodule:: recpack.splitters

.. contents:: Table of Contents
    :depth: 2

Scenarios
------------------

.. autosummary::
    :toctree: generated/

    Scenario

    StrongGeneralization
    WeakGeneralization
    Timed
    StrongGeneralizationTimed
    StrongGeneralizationTimedMostRecent
    LastItemPrediction

Splitters
------------

.. autosummary::
    :toctree: generated/

    splitters.UserSplitter
    splitters.FractionInteractionSplitter
    splitters.TimestampSplitter
    splitters.StrongGeneralizationSplitter
    splitters.UserInteractionTimeSplitter
    splitters.MostRecentSplitter

"""

from recpack.scenarios.scenario_base import Scenario

from recpack.scenarios.last_item_prediction import LastItemPrediction
from recpack.scenarios.strong_generalization_timed_most_recent import StrongGeneralizationTimedMostRecent
from recpack.scenarios.strong_generalization_timed import StrongGeneralizationTimed
from recpack.scenarios.strong_generalization import StrongGeneralization
from recpack.scenarios.timed import Timed
from recpack.scenarios.weak_generalization import WeakGeneralization
