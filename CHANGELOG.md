# CHANGELOG

All notable changes to this project will be documented in this file.  
_Maintainer | Lien Michiels | lien.michiels@froomle.com_
_Maintainer | Robin Verachtert | robin.verachtert@froomle.com_

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED]

### Breaking Changes
* __scenarios__
    * Parameters of the `WeakGeneralization` scenario have been changed to now only accept a single `frac_data_in` parameter. Which makes sure the validation task is as difficult as the test tasks.
    * Training data structure has changed. Rather than a single training dataset, we use two training datasets. `validation_training_data` is used to train models while optimising their parameters, and evaluation happens on the validation data. `full_training_data` is the union of training and validation data on which the final model should be trained during evaluation on the test dataset.

* __pipelines__
    * PipelineBuilder's `set_train_data` function has been replaced with `set_full_training_data` and `set_validation_training_data` to set the two separate training datasets.
    * Added `set_data_from_scenario` function to `PipelineBuilder`, which makes setting data easy based on a split scenario.

* Several module changes were made:
    * splitters module was restructured. Base module is called scenarios. Inside scenarios the splitters submodule contains the splitters functionality.
    * data module was removed, and the submodules were turned into modules.  (`dataset` and `matrix`)

* __datasets__:
    * Default min rating for `Movielens25M` was changed from 1 to 4 similar to most paper preprocessing of the dataset.

* __metrics__:
    * `DiscountedCumulativeGainK` is renamed to `DCGK`
    * `NormalizedDiscountedCumulativeGainK` is renamed to `NDCGK`

### Bugfixes
* __splitters.scenarios__
    * Improved tests for WeakGeneralization scenario to confirm mismatches between train and validation_in are only due to users with not enough items and not a potential bug.
    * Changed order of checks when accessing validation data, so that error when no validation data is configured, is thrown first, rather than the error about split first.

* __algorithms.experimental.time_decay_nearest_neighbour__
    * Improved computation speed of the algorithm by changing typing
    * Added `decay_interval` parameter to improve performance.

### Additions
* __algorithms__
    * Standardised train and predict input validation functions have been added, and used to make sure inputs are InteractionMatrix objects when needed, and contain timestamps when needed.
    * Added `TARSItemKNNLiu` and `TARSItemKNN` algorithms, which compute item-item similarities based on a weighted matrix based on the age of events.
        * `TARSItemKNNLiu` algorithm as defined in Liu, Nathan N., et al. "Online evolutionary collaborative filtering."

    * Added `STAN` algorithm, presented in Garg, Diksha, et al.
    "Sequence and time aware neighborhood for session-based recommendations: Stan". 
    This is a session KNN algorithm that takes into account order and time difference between sessions and interactions.
    * Implemented `NeuMF` as defined in He, Xiangnan, et al. "Neural collaborative filtering." 

* __data.matrix__
    * Added `last_timestamps_matrix`  property which creates a csr matrix, with the last timestamp as nonzero values.

* __data.datasets__
    * Added `AdressaOneWeek` dataset.

* __preprocessing.filters__
    * Added MaxItemsPerUser filter to remove users with extreme amounts of interactions from a dataframe.
    * Added the `SessionDataFramePreprocessor` which cuts user histories into sessions while processing data into InteractionMatrices.

* __pipelines__
    * Added `get_metrics_dataframe` function, which returns the metrics as a dataframe, rather than a dict.

## [0.2.2] - ![](https://img.shields.io/date/1643025589.svg?label=2022-01-24)

### Additions
* __data.datasets__
    * Added CosmeticsShop for https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop
    * Added RetailRocket for https://www.kaggle.com/retailrocket/ecommerce-dataset
    * Added parameters to dummy dataset to define the output expectations.
* __algorithms.baseline__
    * Added boolean parameter `use_only_interacted_items` to Random baseline, selects if all items should be used, or only those interacted with in the training dataset.
* __splitters.scenarios__
    * Added parameter `n_most_recent` to `NextItemPrediction` class to limit test_in data to only the N most recent interactions of each user.

### Changes
* __algorithms.wmf__
    * Refactored WeightedMatrixFactorization
        * It now uses PyTorch operations instead of NumPy (for GPU speedups).
        * It now processes batches of users and items instead of individual users and items.
* __splitters.scenarios__
    * Renamed `NextItemPrediction` to `LastItemPrediction` class, kept `NextItemPrediction` as a copy with deprecation warning. 

### Bugfixes
* __splitters.scenarios__
    * Fixed bug in `NextItemPrediction` scenario:
        * if validation was specified, test_in data contained 1 too few interactions

## [0.2.1] - ![](https://img.shields.io/date/1634291547.svg?label=2021-10-15)
### Dependency Update
* Removed dependency on numba
    * Removed numba decorators in shared account implementation.
      It's potentially slower now, which we don't consider a problem since it is in the experimental module.

### Additions
* __splitters.scenarios__
    * You can set the seed for scenarios with random components. This allows exact recreation of splits for reproducability.
* __pipelines.pipeline__
    * `optimisation_results` property added to Pipeline to allow users to inspect the results for the different hyperparameters that were tried.
* __data.datasets__
    * Added DummyDataset for easy testing purposes.

## [0.2.0] - ![](https://img.shields.io/date/1630311485.svg?label=2021-8-30)

### Additions
* __algorithms.nearest_neighbour__
    * Add ItemPNN, it's ItemKNN, but instead of selecting the top K neighbours, they are sampled from a uniform, softmax-empirical or empirical distribution of item similarities.
    * Add a few options to ItemKNN:
        * pop_discount: Discount relationships with popular items to avoid a large popularity bias
        * sim_normalize: Renamed normalize to sim_normalize. Normalize rows in the similarity matrix to counteract
        artificially large similarity scores when the predictive item is rare, defaults to False.
        * normalize_X: Normalize rows in the interaction matrix so that the contribution of
        users who have viewed more items is smaller
* __data.datasets__
    * `filename` parameter now has a default value for almost all datasets.
    * After initializing a dataset, the code will make sure the specified path exists, and create directories if they were missing.

### Breaking changes:

* __Datasets__:
    * The filename parameter behaviour has changed.
    This parameter used to expect the full path to the file.
    It now expects just the filename, the directory is specified using `path`.

* __preprocessing.preprocessors__:
    * Removed the `USER_IX` and `ITEM_IX` members from DataframePreprocessor. You should use `InteractionMatrix.USER_IX` and `InteractionMatrix.ITEM_IX` instead.

* __util__:
    * `get_top_K_values` and `get_top_K_ranks` parameter `k` changed to `K` so it is in line with rest of Recpack.

## [0.1.2] - ![](https://img.shields.io/date/1627975447.svg?label=2021-8-3)
* Added Gru4Rec algorithms
    * GRU4RecNegSampling
    * GRU4RecCrossEntropy
* added new loss functions:
    * bpr_max_loss
    * top_1_loss
    * top_1_max_loss
* Added sequence batch samplers
* Cleanup of Torch class interface
* Added `predict_topK` parameter to TorchMLAlgorithm baseclass and all children. This parameter is used to cut predictions in case a dense user x item matrix is too large to fit in memory.
* Updated dependencies to support ranges rather than fixed versions.

## [0.1.1] - ![](https://img.shields.io/date/1626758338.svg?label=2021-7-20)
* Added option to install recpack from gitlab pypi repository.
    * See README for instructions.
## [0.1.0] - ![](https://img.shields.io/date/1626354902.svg?label=2021-07-15))

* Very first release of Recpack
* Contains Tested and documented code for:
    * Preprocessing
    * Scenarios
    * Algorithms
    * Metrics
    * Postprocessing
* Contains pipelines for running experiments.
