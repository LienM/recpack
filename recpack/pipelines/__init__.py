"""Pipelines for easy running of experiments.

.. currentmodule:: recpack.pipelines

.. autosummary::
    :toctree: generated/

    PipelineBuilder
    Pipeline

    registries.AlgorithmRegistry
    registries.MetricRegistry


In order to simplify running an experiment,
the pipelines module contains a pipeline which makes sure the necessary steps
are performed in the right order.
To define a pipeline, you should use the PipelineBuilder class,
which makes it possible to construct the pipeline with a few intuitive functions.

An example of usage is::

    import recpack.pipelines

    # Construct pipeline_builder and add data.
    # Assumes a scenario has been created and split before.
    pipeline_builder = recpack.pipelines.PipelineBuilder('demo')
    pipeline_builder.set_data_from_scenario(scenario)

    # We'll have the pipeline optimise the K parameter from the given values.
    pipeline_builder.add_algorithm('ItemKNN', grid={'K': [100, 200, 400, 800]})

    # Add NDCG and Recall to be evaluated at 10, 20, 50 and 100
    pipeline_builder.add_metric('NDCGK', [10, 20, 50, 100])
    pipeline_builder.add_metric('RecallK', [10, 20, 50, 100])

    # Construct pipeline
    pipeline = pipeline_builder.build()

    # Run pipeline, will first do optimisation, and then evaluation
    pipeline.run()

    # Get the metric results.
    # This will be a dict with the results of the run.
    # Turning it into a dataframe makes reading easier
    pd.DataFrame.from_dict(pipeline.get_metrics())


If you want to use the pipelines with your own algorithms or metrics,
you should register them using the :data:`ALGORITHM_REGISTRY` and :data:`METRIC_REGISTRY` respectively.
For info on the functions see :class:`registries.AlgorithmRegistry` and :class:`registries.MetricRegistry`.

Example to register an algorithm::

    from recpack.pipelines import ALGORITHM_REGISTRY
    from recpack.algorithms import ItemKNN

    # Create a new algorithm, that is just a copy of ItemKNN
    class NewAlgorithm(ItemKNN):
        pass

    ALGORITHM_REGISTRY.register('NewAlgorithm', NewAlgorithm)

    # Construct a NewAlgorithm object from the registry
    algo = ALGORITHM_REGISTRY.get('NewAlgorithm')(K=20)


Example to register a metric::

    from recpack.pipelines import METRIC_REGISTRY
    from recpack.algorithms import Recall

    # Define a new metric (that is just a copy of Recall)
    class NewMetric(Recall):
        pass

    METRIC_REGISTRY.register('NewMetric', NewMetric)

    # Construct a NewMetric object with parameter K=20
    algo = METRIC_REGISTRY.get('NewMetric')(K=20)
"""

from recpack.pipelines.pipeline import Pipeline

from recpack.pipelines.registries import ALGORITHM_REGISTRY, METRIC_REGISTRY

from recpack.pipelines.pipeline_builder import PipelineBuilder
