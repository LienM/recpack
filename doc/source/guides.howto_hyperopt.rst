.. _guides-hyperopt:

How to use Hyperopt to Optimise Hyperparameters in a RecPack Pipeline
=====================================================================

The hyperopt library provides access to Tree of Parzen Estimator optimisation, which is more efficient than performing grid search over large search spaces.
For the full documentation of the package, check out their `documentation <http://hyperopt.github.io/hyperopt/>`_.

A RecPack Pipeline can optimise hyperparameters both using a traditional GridSearch or HyperOpt's Tree of Parzen Estimator.
In this guide, we will explain how to configure hyperparameter optimisation using HyperOpt. 

Optimisation Class
------------------
.. currentmodule:: recpack.pipelines

In order to define hyperparameter optimisation with hyperopt for an algorithm, use the :class:`HyperoptInfo` class, which accepts 3 parameters:

- ``space``: Defines per parameter the range of values to sample from.
- ``timeout``: Optional parameter which allows you to stop optimisation after a fixed amount of time.
- ``max_evals``: Optional parameter which specifies how many parameter combinations to try while optimising.

When ``timeout`` and ``max_evals`` are both specified, whichever threshold is first reached, will be used.

Configuring the Search Space
----------------------------
The most important part of using hyperopt, is to specify the search space correctly.
Make sure that the parameter ranges and types are correct to get the best possible results.

The search space is either a Python ``dict``, with as keys the hyperparameters to optimize and as value a hyperopt `parameter expression <https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions>`_,
or a ``hp.choice`` parameter expression when you use conditional parameters.
A conditional parameter is a parameter which should only be set if another parameter has a specific value.

The most likely to use parameter expressions are:

- ``hp.choice(label, options: List)``: Samples one of the specified options.
  You can also use this to handle conditional parameters, by including a parameter expression in the ``options`` list. 
  More information can be found in the examples below.
  RecPack only supports 1 level of conditional parameters.
- ``hp.randint(label, upper)``:  Generates a random number between 0 and ``upper``.
- ``hp.uniform(label, low, high)``: Generates a random value between ``low`` and ``high``, uniformly distributed.
- ``hp.loguniform(label, low, high)``: Generates a random value between ``exp(low)`` and ``exp(high)``, drawn according to ``exp(uniform(low, high))``. 
  Note, that the ``low`` and ``high`` parameters are both exponents and not the boundaries between which the sampled values will lie.
- ``hp.normal(label, mu, sigma)``: Generates a random value, normally distributed with mean ``mu`` and standard deviation ``sigma``.

Examples
--------

The following configuration for optimisation of the ItemKNN algorithm tells the optimiser to sample K values between 50 and 1000, 
and consider both True and False for both normalization options. 
The optimiser is given 50 evaluations, or 1 day of optimisation time, whichever comes first. ::
    
    from hyperopt import hp
    import numpy as np
    from recpack.algorithms import ItemKNN

    HOUR = 3600
    DAY = 24*HOUR

    HyperoptInfo(
        {
            "K": hp.loguniform("K", np.log(50), np.log(1000)),
            "normalize_X": hp.choice("normalize_X", [True, False]),
            "normalize_sim": hp.choice("normalize_sim",[True, False]),
        },
        timeout=DAY,
        max_evals=50,
    )


ItemKNN has a parameter ``pop_discount``, which should only be used when the similarity measure is ``conditional_probability``.
Hyperopt allows nested parametrization for this case, by specifying nested parameter configurations inside the ``hp.choice`` definition.
To have hyperopt explore ``pop_discount`` values only when the right similarity measure is used, we define the ``OptimisationInfo`` as below.
Use the top level ``hp.choice('similarity', ...)`` instead of a dict as used in the previous example. 
Both options of the choice parameter should be dictionaries, structured similarly to a non conditional setup.
When sampling parameters, hyperopt will first choose which of the two options to use, and then sample values for each parameter in that option's dictionary.:: 

    HyperoptInfo(
        hp.choice(
            'similarity', [
                {
                    'similarity': 'conditional_probability', 
                    'pop_discount': hp.uniform('pop_discount', 0, 1),
                    "K": hp.loguniform("cp_K", np.log(50), np.log(1000)),
                    "normalize_X": hp.choice("cp_normalize_X", [True, False]),
                    "normalize_sim": hp.choice("cp_normalize_sim",[True, False]),
                }, 
                {
                    'similarity': 'cosine',
                    "K": hp.loguniform("c_K", np.log(50), np.log(1000)),
                    "normalize_X": hp.choice("c_normalize_X", [True, False]),
                    "normalize_sim": hp.choice("c_normalize_sim",[True, False]),
                }
            ]
        ),
        timeout=DAY,
        max_evals=50,
    )

.. note::
    Note that you have to use different identifiers in the parameter expression if parameters are shared between options (``"normalize_X"`` for example).

.. warning:: 
    When using nested parameters, do not use more than 1 nesting level.
    Hyperopt allows you to specify any level of nested parameters, however, 
    the nested parameters remain nested in the parameter dictionary, and can not be mapped to parameters in the ``Algorithm`` class.


Running a Pipeline
------------------

The configured optimisation information should be used in a pipeline to optimise the hyperparameters of the algorithm.

:: 

    from hyperopt import hp
    import numpy as np

    from recpack.algorithms import ItemKNN
    from recpack.datasets import DummyDataset
    from recpack.pipelines import PipelineBuilder, HyperoptInfo
    from recpack.scenarios import WeakGeneralization

    HOUR = 3600
    DAY = 24*3600

    optimisation_info = HyperoptInfo(
        hp.choice(
            'similarity', [
                {
                    'similarity': 'conditional_probability', 
                    'pop_discount': hp.uniform('pop_discount', 0, 1),
                    "K": hp.loguniform("cp_K", np.log(50), np.log(1000)),
                    "normalize_X": hp.choice("cp_normalize_X", [True, False]),
                    "normalize_sim": hp.choice("cp_normalize_sim",[True, False]),
                }, 
                {
                    'similarity': 'cosine',
                    "K": hp.loguniform("c_K", np.log(50), np.log(1000)),
                    "normalize_X": hp.choice("c_normalize_X", [True, False]),
                    "normalize_sim": hp.choice("c_normalize_sim",[True, False]),
                }
            ]
        ),
        timeout=DAY,
        max_evals=50,
    )

    im = DummyDataset().load()
    scenario = WeakGeneralization(validation=True)
    scenario.split(im)

    pb = PipelineBuilder()
    # Use the configured search space when optimising the ItemKNN parameters
    pb.add_algorithm(ItemKNN, optimisation_info=optimisation_info)
    pb.set_data_from_scenario(scenario)
    # Set NDCG@10 as the optimisation metric.
    # Since it is a metric to be maximized, the loss will be the negative of the NDCG, such that hyperopt can minimize it.
    pb.set_optimisation_metric('NDCGK', 10)
    pb.add_metric('NDCGK', 10)

    pipe = pb.build()
    pipe.run()

