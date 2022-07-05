Welcome to Recpack's documentation!
===================================

RecPack is an experimentation toolkit for top-N recommendation using implicit feedback data written in Python, with a familiar interface and clear documentation.
Its goals is to support researchers who advance the state-of-the-art in top-N recommendation to write reproducible and reusable experiments.
RecPack comes with a range of different datasets, recommendation scenarios, state-of-the-art baselines and metrics.
Wherever possible, RecPack sets sensible defaults. For example, hyperparameters of all recommendation algorithms included in RecPack are initialized to the best performing settings found in the original experiments.
The design of RecPack is heavily inspired by the interface of scikit-learn, a popular Python package for classification, regression and clustering tasks. 
Data scientists who are familiar with scikit-learn will already have an intuitive understanding of how to work with RecPack.
On top of this, RecPack was developed with a production mindset: All contributions are rigorously reviewed and tested.
The RecPack maintainers strive to maintain a test coverage of more than ninety percent at all times.

Using RecPack, you can:

* **Quickly implement new algorithms** by using one of RecPack's abstract algorithm base classes. These base classes represent popular recommendation paradigms such as matrix factorization and deep learning.
* **Compare their algorithms against state-of-the-art baselines** in several different recommendation scenarios using a variety of different performance metrics.
* **Tune hyperparameters** of both baselines and your own implementations with minimal data leakage. 

A typical experimentation pipeline for top-N recommendation is shown in the Figure below.
RecPack provides a dedicated module to support each step. For the detailed documentation on each module, check out their documentation pages below.

.. image:: ./img/diagram.png


.. toctree::
   :maxdepth: 2

   recpack.datasets
   recpack.preprocessing
   recpack.matrix
   recpack.scenarios
   recpack.algorithms
   recpack.postprocessing
   recpack.metrics
   recpack.pipelines
   guides

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
