# RecPack
RecPack is an experimentation toolkit for top-N recommendation, using implicit feedback data written in Python, with a familiar interface and clear documentation. Its goal is to support researchers who advance the state-of-the-art in top-N recommendation to write reproducible and reusable experiments. RecPack comes with a range of different datasets, recommendation scenarios, state-of-the-art baselines and metrics. Wherever possible, RecPack sets sensible defaults. For example, hyperparameters of all recommendation algorithms included in RecPack are initialized to the best performing settings found in the original experiments. The design of RecPack is heavily inspired by the interface of scikit-learn, a popular Python package for classification, regression and clustering tasks. Data scientists who are familiar with scikit-learn will already have an intuitive understanding of how to work with RecPack. On top of this, RecPack was developed with a production mindset: All contributions are rigorously reviewed and tested. The RecPack maintainers strive to maintain a test coverage of more than ninety percent at all times.


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

If you use RecPack for research purposes, please cite the [paper](https://doi.org/10.1145/3523227.3551472).

```
@inproceedings{10.1145/3523227.3551472,
    author = {Michiels, Lien and Verachtert, Robin and Goethals, Bart},
    title = {RecPack: An(Other) Experimentation Toolkit for Top-N Recommendation Using Implicit Feedback Data},
    year = {2022},
    isbn = {9781450392785},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3523227.3551472},
    doi = {10.1145/3523227.3551472},
    booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
    pages = {648–651},
    numpages = {4},
    location = {Seattle, WA, USA},
    series = {RecSys '22}
}
```

## License

RecPack, An Experimentation Toolkit for Top-N Recommendation
Copyright (C) 2020  Froomle N.V.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

A copy of the GNU Affero General Public License should be distributed along with this program.
If not, see <http://www.gnu.org/licenses/>.

Address Froomle: Posthofbrug 6-8, 2600 Antwerpen-Berchem
Contact Froomle: robin.verachtert[at]froomle.com
