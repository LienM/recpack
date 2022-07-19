.. _guides-quickstart:

Quickstart
==========

Installation
------------
The latest version can be installed from `Pypi <https://pypi.org/project/recpack/>`_. ::

    pip install --upgrade recpack


Basic Usage
-----------

RecPack was set up to allow easy experimentation with various algorithms,
using different datasets or scenarios.

When running an experiment, the first step is to load and preprocess a dataset.
RecPack provides a set of classes to handle some of the most used datasets. ( :mod:`recpack.datasets` ) ::

    from recpack.datasets import MovieLens25M
    dataset = MovieLens25M(path='/path/to/file')
    dataset.fetch_dataset()


Here we select the MovieLens25M dataset. We explicitly set a path to where the dataset should be stored.
When calling ``.fetch_dataset()`` the code will check if a file containing the dataset is present and download it if it is not.

Now that we have the Dataset, we want to load it and preprocess it so that it is ready for use in the experiment.
RecPack Datasets have default preprocessing steps defined such that they follow frequently used preprocessing from literature.
In the case of MovieLens25M the rating data will be turned into binary data, by keeping only interactions with rating 4 or higher.
Further, users with less than 3 interactions are removed as well as items with less than 5 interactions.

The preprocessing step also creates a consecutive user and item index, since it is common to represent the data as a user-item interaction matrix.

To download the dataset and preprocess, call::

    interaction_matrix = dataset.load()

The result is an :class:`recpack.matrix.InteractionMatrix` object.
This class is a wrapper around the data that provides frequently used views of the interaction data.

For example, ``interaction_matrix.values`` returns a csr_matrix, where users are rows and items are column and the value is the number of interactions this user had with this item.
Similarly, ``interaction_matrix.sorted_item_history`` returns an iterator of the sorted item histories per user.

Next step in an experiment is to separate training, validation and test data to avoid data leakage.
RecPack handles this using the :mod:`recpack.scenarios` module. It includes some of the most common experimentation scenarios.

Choosing the right scenario is crucial in getting representative experimental results, so make sure to check the documentation of the scenarios to make sure they match your intent.

.. currentmodule:: recpack.scenarios

::

    from recpack.scenarios import WeakGeneralization
    scenario = WeakGeneralization(0.75, validation=True)

Here we use :class:`WeakGeneralization` as it is a frequently used scenario for MovieLens25M.
We set ``frac_data_in`` to 0.75. As a result, 25% of each user's interactions are used as test target data and the remaining 75% are used as both history and full training data.
Because we also request validation data (so we can perform parameter optmisation), the full training data will be further split 75/25 into validation train (and history) and validation target data.

Now that we have our data preprocessed and divided into train, validation and test datasets we can set up a pipeline to optimize our hyperparameters and obtain performance metrics.
Rather than linking up the remaining components (Algorithm, Optimisation, Postprocessing, Metrics) separately, we will use the recpack Pipeline.
To facilitate building a pipeline we use :class:`recpack.pipelines.PipelineBuilder`. ::

    from recpack.pipelines import PipelineBuilder
    
    builder = PipelineBuilder()

To use our training, validation and test dataset we pass the scenario to the builder ::

    builder.set_data_from_scenario(scenario)

Next we select the algorithms to use in the experiment and their hyper-parameters.
To optimise hyper-parameters, they are specified in the ``grid`` option, specifying per parameter the values to try.
These will be combined into a full grid such that every combination of parameters is tried. ::

    builder.add_algorithm('Popularity') # No real parameters to optimise
    builder.add_algorithm('ItemKNN', grid={
        'K': [100, 200, 500],
        'similarity': ['cosine', 'conditional_probability'],
    })

Because we need to optimise parameters, we should also tell the pipeline which metric to use for evaluating them.
We choose to use the traditional ranking metric ``NDCG@10``. ::    

    builder.set_optimisation_metric('NDCGK', K=10)

Finally we need to select which metrics we want the pipeline to evaluate our algorithms on.
We again use NDCG but now at various K values for more insight. We also use Coverage to see how much of the item catalog is recommended.::

    builder.add_metric('NDCGK', K=[10, 20, 50])
    builder.add_metric('CoverageK', K=[10, 20])

Time to build and run the pipeline::

    pipeline = builder.build()
    pipeline.run()

Depending on your dataset and selected algorithms running the pipeline can take a while.
When it is done, you can get the metrics for each algorithm using ::

    pipeline.get_metrics()

Should you want to see what effects parameter optimisation had, you can also inspect the optimisation results using ::

    pipeline.optimisation_results

You should now be able to set up your own experiments.

Using your own dataset
----------------------------

.. currentmodule:: recpack.preprocessing.preprocessors

RecPack supports a large selection of datasets, but you might still have a dataset for which we don't have a Dataset class.
In this case you can load the data manually into a pandas DataFrame, and use the :class:`DataFramePreprocessor`. ::

    from recpack.preprocessing.preprocessors import DataFramePreprocessor
    from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem

    proc = DataFramePreprocessor(item_ix='item_id', user_ix='user_id', timestamp_ix='ts')
    proc.add_filter(MinUsersPerItem(5, item_ix='item_ix', user_ix='user_id'))
    proc.add_filter(MinItemsPerUser(5, item_ix='item_ix', user_ix='user_id'))

    # Assuming you have loaded a dataframe called df
    interaction_matrix = proc.process(df)

In this example we preprocess an existing dataframe ``df`` into an interaction matrix.
Here we've assumed the user identifiers are stored in column ``'user_id'`` in the original dataframe and item identifiers in ``'item_id'``. Timestamps are optional, but our example dataset includes them in column ``ts``. 
and timestamps in column ``'ts'``
We filter out users with fewer than 5 interactions, and items with fewer than 5 interactions.

Creating your own algorithm
---------------------------
In the previous examples we always used preimplemented algorithms, however it is often most interesting to be able to test your own new algorithms against the baselines.
For this check out the :ref:`guides-algorithms` guide, which will detail this process.