# RecPack
RecPack is an experimentation toolkit for top-N recommendation using implicit feedback data written in Python, with a familiar interface and clear documentation. Its goals is to support researchers who advance the state-of-the-art in top-N recommendation to write reproducible and reusable experiments. RecPack comes with a range of different datasets, recommendation scenarios, state-of-the-art baselines and metrics. Wherever possible, RecPack sets sensible defaults. For example, hyperparameters of all recommendation algorithms included in RecPack are initialized to the best performing settings found in the original experiments. The design of RecPack is heavily inspired by the interface of scikit-learn, a popular Python package for classification, regression and clustering tasks. Data scientists who are familiar with scikit-learn will already have an intuitive understanding of how to work with RecPack. On top of this, RecPack was developed with a production mindset: All contributions are rigorously reviewed and tested. The RecPack maintainers strive to maintain a test coverage of more than ninety percent at all times.


## Installation

All released versions of recpack are published on [Pypi](https://pypi.org/project/recpack/), and can be installed using:

`pip install recpack`

## Documentation
Documentation and tutorials can be found at https://recpack.froomle.ai

## Usage
RecPack provides a framework for experimentation with recommendation algorithms. 
It comes pre-packed with a number of commonly used evaluation scenarios (scenarios),
evaluation metrics (metrics) and state-of-the-art algorithm implementations (algorithms).
New algorithms and evaluation scenarios can be added easily, by subclassing the appropriate base classes. 
A number of lower level data splitters are provided that can be used to build up more complex evaluation scenarios.

Users can choose to use the Pipeline interface or manually connect components for running experiments. 
The Pipeline interface is recommended for easy comparison between algorithms. For optimal flexibility you should manually connect components. 

For details and examples, check out the [quickstart documentation](recpack.froomle.ai/guides.quickstart.html)

