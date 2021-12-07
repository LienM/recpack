# CHANGELOG

All notable changes to this project will be documented in this file.  
_Maintainer | Lien Michiels | lien.michiels@froomle.com_
_Maintainer | Robin Verachtert | robin.verachtert@froomle.com_

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED]
* Added boolean parameter `use_only_interacted_items` to Random baseline, selects if all items should be used, or only those interacted with in the training dataset.
* Fixed bug in NextItem prediction scenario
    * if validation was specified, test_in data contained 1 too few interactions
* Added parameter `n_most_recent` to `NextItemPrediction` class to limit test_in data to only the N most recent interactions of each user.
* Renamed NextItemPrediction to LastItemPrediction class, kept NextItemPrediction as a copy with deprecation warning. 

* __algorithms.wmf__
    * Refactored WeightedMatrixFactorization
        * It now uses PyTorch operations instead of NumPy (for GPU speedups).
        * It now processes batches of users and items instead of individual users and items.

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
