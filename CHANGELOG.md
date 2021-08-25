# CHANGELOG

All notable changes to this project will be documented in this file.  
_Maintainer | Lien Michiels | lien.michiels@froomle.com_
_Maintainer | Robin Verachtert | robin.verachtert@froomle.com_

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Additions
* __Datasets__
    * `filename` parameter for almost all datasets received a default value
    * After initializing a dataset, the code will make sure the specified path exists, creating directories that were missing.
### Breaking changes:

* __Datasets__:
    * The filename parameter behaviour has changed. 
    This parameter used to expect the full path to the file. 
    It now is expected to be just the filename. 
    The directory where the file is stored, or where the downloaded file should be put is configured using the `path` parameter.

## [0.1.3]
* Removed the `USER_IX` and `ITEM_IX` members from DataframePreprocessor.

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
