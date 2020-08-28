# RecPack
Python package for easy experimentation with recsys algorithms

## Installation

1. Clone repository
2. In repository run `pip install .`

## Usage
RecPack provides a framework for experimentation with recommendation algorithms. 
It comes pre-packed with a number of commonly used evaluation scenarios (splitters),
evaluation metrics (metrics) and state-of-the-art algorithm implementations (algorithms).
New algorithms and evaluation scenarios can be added easily, by subclassing the appropriate base classes. 
A number of lower level data splitters are provided that can be used to build up more complex evaluation scenario's.

Users can choose between the Experiment and Pipeline interface for running experiments. 
Use Experiment if you want full control over hyperparameters and want to optimize a single algorithm. 
Pipelines allow easy comparison between algorithms. 


### Load and preprocess data

At the core of RecPack is the DataM object. 
This DataM object represents your data. 
It provides a number of easy, heavily optimized operations for indexing into and slicing your data. 

Currently RecPack provides an easy to use Preprocessor to transform a Pandas DataFrame into a DataM.  
If for some reason you are unable to load your data into a Pandas DataFrame, you can extend the Preprocessor base class. 
Under the hood, this DataFramePreprocessor maps user and item indices to a sequence of consecutive integers.
This is required by many algorithms to allow for easy computation and avoid singularity. 


```python
import pandas as pd

from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.data.data_matrix import DataM

pv_frame = pd.read_csv("pageviews.csv")
pur_frame = pd.read_csv("purchases.csv")

prep = DataFramePreprocessor("itemId", "userId")
pv_m, pur_m = prep.process(pv_frame, pur_frame) # Transform to DataM objects.

```

Important to note is that when you use multiple data inputs as in the example above, you want to process them together, so that all resulting DataM objects have the same size. 

### Using an algorithm

RecPack provides a number of state-of-the-art algorithm implementations. 

To get the list of all algorithms use:
```python
import recpack.algorithms
recpack.algorithms.algorithm_registry.list()
```
This will give you a list of algorithms available for immediate use.

To create an algorithm instance use one of:

```python
import recpack.algorithms
from recpack.algorithms import EASE

ease1 = recpack.algorithms.algorithm_registry.get('ease')(lambda=1000)

ease2 = EASE(lambda=1000)
```
For information on which parameters can be used for which algorithm see their individual docstrings.

#### Implementing a new algorithm
Should you want to implement a new algorithm which we do not yet support you should create a subclass of the `Algorithm` base class

Here is the code for a popularity algorithm

```python
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.base import Algorithm

class Popularity(Algorithm):

    def __init__(self, K=200):
        self.sorted_scores_ = None
        self.K = K

    def fit(self, X):
        items = list(X.nonzero()[1])
        self.sorted_scores_ = Counter(items).most_common()

    def predict(self, X):
        """For each user predict the K most popular items"""
        score_list = np.zeros((X.shape[1]))
        for i, s in self.sorted_scores[:self.K]:
            score_list[i] = s
        return csr_matrix(np.repeat([score_list], X.shape[0], axis=0))

    @property
    def name(self):
        return f"popularity_{self.K}"
```

// TODO Elaborate on conventions when developing new algorithms 

### Selecting a scenario
The splitter class will take the input data and split it into 3 data objects:
* train
* validation
* test

These will be used further in the pipeline to train models, and evaluate the models.

```python
import recpack.splits
# Construct a splitter object which uses strong generalization to split the data into
# three data objects. Train will contain 50% of the users, validation 20% and test 30%
# The seed parameter is useful for creating reproducible results.
splitter = recpack.splits.StrongGeneralizationSplit(0.5, 0.2, seed=42)
```

### Selecting an evaluator
The evaluator class takes the 3 data objects and generate in and output matrices.
The input matrix will be passed as input to the predict method of the algorithm, the out matrix is the true output and will be used by the metrics.

```python
import recpack.evaluate
# Construct a fold in evaluator.
# this evaluator will take the test data, and for each user add 40% of their interactions
# to the in_ matrix and 60% to the out_ matrix.
# Seed again for reproducability.
evaluator = recpack.evaluate.FoldInPercentageEvaluator(0.4, seed=42)
```

### Creating the pipeline
When creating the pipeline you will connect all the components and select metrics which the pipeline will compute.
For the metrics we support three at the moment: `NDCG`, `Recall`, `MRR`. Each of these will be computed for each K value passed in the K_values list parameter.

```python
import recpack.pipelines
# Construct a pipeline which computes NDCG and Recall @ 10, 20, 50 and 100
# Only a single algorithm is evaluated, but you could select multiple algorithms to be evaluated at the same time.
p = recpack.pipelines.Pipeline(splitter, [algo], evaluator, ['NDCG', 'Recall'], [10,20,50,100])

# Get the metric results.
# This will be a dict with the results of the run.
p.get()
```

## More information on the pipeline
![alt text](images/pipeline.png "pipeline structure")
