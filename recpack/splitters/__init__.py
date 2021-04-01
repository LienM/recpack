"""

The splitters module contains many of the most commonly encountered evaluation
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
you can use these splitters to help you build it from the ground up.

.. currentmodule:: recpack.splitters

.. contents:: Table of Contents
    :depth: 2

Scenarios
------------------

.. autosummary::
    :toctree: generated/

    scenario_base.Scenario

    scenarios.StrongGeneralization
    scenarios.WeakGeneralization
    scenarios.Timed
    scenarios.StrongGeneralizationTimed
    scenarios.StrongGeneralizationTimedMostRecent

Splitters
------------

.. autosummary::
    :toctree: generated/

    splitter_base.UserSplitter
    splitter_base.FractionInteractionSplitter
    splitter_base.TimestampSplitter
    splitter_base.StrongGeneralizationSplitter
    splitter_base.UserInteractionTimeSplitter
    splitter_base.MostRecentSplitter

"""
