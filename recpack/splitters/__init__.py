"""The splitter module is responsible for splitting data into train,
validation and test data.

Recpack provides preimplemented scenarios for well known splitting techniques.
Should you want to implement a specific not yet implemented scenario,
data splitters are provided to perform frequently used splits.

.. currentmodule:: recpack.splitters

.. contents:: Table of Contents
    :depth: 2

Scenarios
------------------
A scenario splits data, into a training dataset, 
a validation dataset split into two folds (input and expected output)
and a test dataset split into two folds (input and expected output) 
used to split data into the train, validation and test sub datasets.

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