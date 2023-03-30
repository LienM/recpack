# CHANGELOG

All notable changes to this project will be documented in this file.  
_Maintainer | Lien Michiels | lien.michiels@froomle.com_
_Maintainer | Robin Verachtert | robin.verachtert@froomle.com_

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED]

## Additions
* __algorithms__
    * `TimeAwareItemKNN` was moved from experimental to the main algorithms package, and renamed to `TARSItemKNNXia` to distinguish it from other time aware item KNN models.
    * Created a base class for time aware variants of `ItemKNN`.
    * Added the following time aware variants of `ItemKNN`:
        * `TARSItenKNNHermann` as Presented in Hermann, Christoph. "Time-based recommendations for lecture materials." EdMedia+ Innovate Learning. Association for the Advancement of Computing in Education (AACE), 2010.
        * `TARSItemKNNVaz` as described in Vaz, Paula Cristina, Ricardo Ribeiro, and David Martins De Matos. "Understanding the Temporal Dynamics of Recommendations across Different Rating Scales." UMAP Workshops. 2013.
        * `SequentialRules` as described in Ludewig, Malte, and Dietmar Jannach. "Evaluation of session-based recommendation algorithms." User Modeling and User-Adapted Interaction 28.4 (2018): 331-390.
        * `TARSItemKNNLee` from Lee, Tong Queue, Young Park, and Yong-Tae Park. "A time-based approach to effective recommender systems using implicit feedback." Expert systems with applications 34.4 (2008): 3055-3062.
        * `TARSItemKNNDing` from Ding, Yi, and Xue Li. "Time weight collaborative filtering." Proceedings of the 14th ACM international conference on Information and knowledge management. 2005.
        * `TARSItemKNNLiu2012`based on Liu, Yue, et al. "Time-based k-nearest neighbor collaborative filtering."

### Bugfixes
* __metrics__
    * Fixed bug in `CalibratedRecallK` results property, which caused an error when requesting this property.
* __datasets__
    * Made Datasets more robust by removing the defaults for `USER_IX`, `ITEM_IX`, `TIMESTAMP_IX`.
    * Fixed bug in DummyDataset that used `num_users` to set `self.num_items`. 
* __matrix__
    * Made `InteractionMatrix` more robust to funky indices.
    * `InteractionMatrix` now verifies if shape matches
* __tests.test_algorithms__
    * Removed unused fixtures and cleaned up namings.

## [0.3.5] - ![](https://img.shields.io/date/1676895426.svg?label=2023-02-20)

### Bugfixes
* Replaced remaining usage of `np.bool` with `bool` to be compatible with numpy `1.24.1`
* __algorithms__
    * `GRU4Rec` fixed bug in training, where it still tried to learn padding tokens.
* __preprocessing__
  * Make sure DataFrames returned in filters and preprocessors are copies, and no DataFrame is ever edited inplace.

### Additions
* Added getting started guide for using Hyperopt optimisation
* __matrix__
    * Added `items_in` function, which selects only those interactions with the given list of items.
    * Added `active_items` and `num_active_items` properties, to get the active items in a dataset.
* __datasets__
    * Added MovieLens100K, MovieLens1M, MovieLens10M and MillionSongDataset (aka TasteProfileDataset).
    * Added Globo dataset
* __pipelines__
    * Added option to use hyperopt to optimise parameters
        * `grid` parameter has been superseded by the `optimisation_info` parameter, which takes an `OptimisationInfo` object. Two relevant subclasses have been defined: `GridSearchInfo` contains a grid with parameters to support grid search, `HyperoptInfo` allows users to specify a hyperopt space, with timeout and or maximum number of evaluations.
    * Extended output of optimisation output, to now also include an `Algorithm` column for easier analysis.

### Changes
* __datasets__
    * CosmeticsShop now loads from zip archive instead of requiring users to manually append files together
* __tests.test_datasets__
  * Split dataset tests into different files for ease of reading.
* __algorithms__
    * `RecVAE` and `MultVAE` implementations were updated to use the `get_batches` function.
* __pipelines__
    * Made `PipelineBuilder` work with `Metric` (without K). Affects only `PercentileRanking`
    * Fixed issue with optimisation selecting the wrong parameters.
* __preprocessing__
  * Made sure DataFrames returned in filters and preprocessors are copies, and no DataFrame is ever edited inplace.
    * Updated `Prod2Vec` and `Prod2VecClustered` to remove similarities from and to unvisited items.

### Breaking Changes
* __metrics__:
    * `.name` property of a metric now returns camelcase name instead of lowercase name used before.
* __splitters__
    * Helper function `yield_batches` and class `FoldIterator` were removed, as they were unused in samples.
      Alternative function `get_batches` from `algorithms.util` should be used instead.

## [0.3.4]

### Bugfixes
* __algorithms__
    * `BPRMF`: 
        * Changed optimizer to Adagrad instead of SGD. 
        * Set maximum on std of embedding initialisation to make sure the initial embedding does not contain too large values.


## [0.3.3] - ![](https://img.shields.io/date/1666253209.svg?label=2022-10-20)

### Bugfixes
* __algorithms__
    * `NMF`: updated to sklearn `alpha_W` and `alpha_H` parameters, but keeping our single `alpha` parameter.

* Removed the `dataclasses` package dependency, this is included by default in python already.
* Added a step to run example notebooks during testing to make sure these do not break.
* Uses DummyDataset for all demo notebooks to avoid long download and runtimes in the example notebooks.

## [0.3.2] - ![](https://img.shields.io/date/1663337481.svg?label=2022-09-16)

### Breaking Changes
* __algorithms__
    * Renamed hyperparameters for clarity
        * Renamed `embedding_size` to `num_components` to follow sklearn parameter names.
        * Renamed `U` parameter to `num_negatives`
        * Renamed `num_neg_samples` to `num_negatives` for consistency.
        * Renamed `J` for warp loss to `num_items`
    * Deprecated parameter `normalize` of ItemKNN has been removed.
* __scenarios__
    * `n_most_recent` parameter for LastItemPrediction is changed to `n_most_recent_in` and default value changed to infinity such that it is inline with the TimedLastItemPrediction scenario.
    * `n` parameter for `StrongGeneralizationTimedMostRecent`scenario has been renamed to `n_most_recent_out`, and behaviour for a negative value has been removed.
* __datasets__
    * renamed `load_dataframe` to `_load_dataframe`, it is now a private member, which should not be called directly.
    * renamed `preprocessing_default` parameter to `use_default_filters` for clarity.

### Additions
* __scenarios__
    * Added default values for parameters of WeakGeneralization and StrongGeneralization.

### Optimisations
* __datasets__
    * Improved performance of load functionality for AdressaOneWeek.


## [0.3.1] - ![](https://img.shields.io/date/1662375599.svg?label=2022-09-05)

### Additions
* __datasets__
    * Changed behaviour of `force=True` on the Adressa dataset, it is now guaranteed to redownload the tar file, and will no longer look for a local tar file. The tar file will also be deleted at the end of the download method.
    * Added `Netflix` dataset to use the Netflix Prize dataset.
* __scenarios__
    * Added `TimedLastItemPrediction` scenario, which is different from LastItemPrediction, in that it only trains its model on data before a certain timestamp, and evaluates only on data after that timestamp, thus avoiding leakage.
* Configured optional extra installs:
    * `pip install recpack[doc]` will install the dependencies to generate documentation
    * `pip install recpack[test]` will install the dependencies to run tests.

### Bugfixes
* Pinned version of sphinx
* __algorithms__
    * Added `validation_sample_size` parameter to the TorchMLAlgorithm base class and all child classes. This parameter allows a user to select only a sample of the validation data in every evaluation iteration. This speeds up the evaluation step after every training epoch significantly.


## [0.3.0] - ![](https://img.shields.io/date/1657025737.svg?label=2022-07-05)

The 0.3.0 release contains large amounts of changes compared to 0.2.2, these changes should make the library more intuitively useable, and better serve our users.
In addition to interface changes and additions we also made a big effort to increase documentation, and improve it's readability.

### Breaking Changes
* __scenarios__
    * Parameters of the `WeakGeneralization` scenario have been changed to now only accept a single `frac_data_in` parameter. Which makes sure the validation task is as difficult as the test tasks.
    * Training data structure has changed. Rather than a single training dataset, we use two training datasets. `validation_training_data` is used to train models while optimising their parameters, and evaluation happens on the validation data. `full_training_data` is the union of training and validation data on which the final model should be trained during evaluation on the test dataset.
* __pipelines__
    * PipelineBuilder's `set_train_data` function has been replaced with `set_full_training_data` and `set_validation_training_data` to set the two separate training datasets.
    * Added `set_data_from_scenario` function to `PipelineBuilder`, which makes setting data easy based on a split scenario.
    * Updated `get_metrics` function, which returns the metrics as a dataframe, rather than a dict.
* Several module changes were made:
    * splitters module was restructured. Base module is called scenarios. Inside scenarios the splitters submodule contains the splitters functionality.
    * data module was removed, and the submodules were turned into modules (`dataset` and `matrix`). The matrix file is split into additional files as well.
* __datasets__:
    * Default min rating for `Movielens25M` was changed from 1 to 4 similar to most paper preprocessing of the dataset.
    * `load_interaction_matrix` function was renamed to `load`
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

* __data.matrix__
    * Added `last_timestamps_matrix`  property which creates a csr matrix, with the last timestamp as nonzero values.

* __data.datasets__
    * Added `AdressaOneWeek` dataset.

* __preprocessing.filters__
    * Added MaxItemsPerUser filter to remove users with extreme amounts of interactions from a dataframe.
    * Added the `SessionDataFramePreprocessor` which cuts user histories into sessions while processing data into InteractionMatrices.

* Notebook with implementation of NeuMF for demonstration purposes.

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

### Breaking changes
* __Datasets__:
    * The filename parameter behaviour has changed.
    This parameter used to expect the full path to the file.
    It now expects just the filename, the directory is specified using `path`.
* __preprocessing.preprocessors__:
    * Removed the `USER_IX` and `ITEM_IX` members from DataframePreprocessor. You should use `InteractionMatrix.USER_IX` and `InteractionMatrix.ITEM_IX` instead.
* __util__:
    * `get_top_K_values` and `get_top_K_ranks` parameter `k` changed to `K` so it is in line with rest of Recpack.


## [0.1.2] - ![](https://img.shields.io/date/1627975447.svg?label=2021-8-3)

### Additions
* Cleanup of Torch class interface
* Updated dependencies to support ranges rather than fixed versions.
* __algorithms__
    * Added GRU4RecNegSampling
    * Added GRU4RecCrossEntropy 
    * Added new loss functions:
        * bpr_max_loss
        * top_1_loss
        * top_1_max_loss
    * Added sequence batch samplers
    * Added `predict_topK` parameter to TorchMLAlgorithm baseclass and all children. This parameter is used to cut predictions in case a dense user x item matrix is too large to fit in memory.


## [0.1.1] - ![](https://img.shields.io/date/1626758338.svg?label=2021-7-20)

* Added option to install recpack from gitlab pypi repository.
    * See README for instructions.


## [0.1.0] - ![](https://img.shields.io/date/1626354902.svg?label=2021-07-15))

* Very first release of Recpack
* Contains tested and documented code for:
    * Preprocessing
    * Scenarios
    * Algorithms
    * Metrics
    * Postprocessing
* Contains pipelines for running experiments.
