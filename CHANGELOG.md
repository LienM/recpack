# CHANGELOG

All notable changes to this project will be documented in this file.  
_Maintainer | Lien Michiels | lien.michiels@froomle.com_
_Maintainer | Robin Verachtert | robin.verachtert@froomle.com_

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - ![](https://img.shields.io/date/1627536503.svg?label=2021-7-29) 
* Added Gru4Rec algorithms 
    * GRU4RecNegSampling
    * GRU4RecCrossEntropy
* added new loss functions:
    * bpr_max_loss
    * top_1_loss
    * top_1_max_loss
* Added sequence batch samplers
* Cleanup of Torch class interface

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
