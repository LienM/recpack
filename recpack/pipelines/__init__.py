"""Pipelines for easy running of experiments.

.. currentmodule:: recpack.pipelines

.. autosummary::
    :toctree: generated/

    pipeline.PipelineBuilder
    pipeline.Pipeline

    pipeline.ALGORITHM_REGISTRY
    pipeline.METRIC_REGISTRY


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
    pipeline_builder.set_train_data(scenario.training_data)
    pipeline_builder.set_test_data(scenario.test_data)
    pipeline_builder.set_validation_data(scenario.validation_data)

    # We'll have the pipeline optimise the K parameter from the given values.
    pipeline_builder.add_algorithm('ItemKNN', grid={'K': [100, 200, 400, 800]})

    # Add NDCG and Recall to be evaluated at 10, 20, 50 and 100
    pipeline_builder.add_metric('NormalizedDiscountedCumulativeGainK', [10, 20, 50, 100])
    pipeline_builder.add_metric('RecallK', [10, 20, 50, 100])

    # Construct pipeline
    pipeline = pipeline_builder.build()

    # Run pipeline, will first do optimisation, and then evaluation
    pipeline.run()

    # Get the metric results.
    # This will be a dict with the results of the run.
    # Turning it into a dataframe makes reading easier
    pd.DataFrame.from_dict(pipeline.get_metrics())
"""

from recpack.pipelines.pipeline import (
    Pipeline,
    PipelineBuilder,
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
)
