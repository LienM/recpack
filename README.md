# RecPack
Python package for easy experimentation with recsys algorithms.

## Installation

1. Clone repository
2. In repository run `pip install .`


## documentation
Documentation and tutorials can be found at https://adrem-recommenders.gitlab.io/recpack/

## Usage
RecPack provides a framework for experimentation with recommendation algorithms. 
It comes pre-packed with a number of commonly used evaluation scenarios (splitters),
evaluation metrics (metrics) and state-of-the-art algorithm implementations (algorithms).
New algorithms and evaluation scenarios can be added easily, by subclassing the appropriate base classes. 
A number of lower level data splitters are provided that can be used to build up more complex evaluation scenarios.

Users can choose between the Pipeline interface, and manually connecting components for running experiments. 
Pipelines are easier to set up comparisons between algorithms, for full control manual linking is available.


### Load and preprocess data

At the core of RecPack is the InteractionMatrix object. 
This InteractionMatrix object represents your data. 
It provides a number of easy, heavily optimized operations for indexing into and slicing your data. 

Currently RecPack provides an easy to use Preprocessor to transform a Pandas DataFrame into a InteractionMatrix.  
If for some reason you are unable to load your data into a Pandas DataFrame, you can extend the Preprocessor base class. 
Under the hood, this DataFramePreprocessor maps user and item indices to a sequence of consecutive integers.
This is required by many algorithms to allow for easy computation and avoid singularity. 


```python
import pandas as pd

from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.data.matrix import InteractionMatrix

pv_frame = pd.read_csv("pageviews.csv")
pur_frame = pd.read_csv("purchases.csv")

prep = DataFramePreprocessor("itemId", "userId")
pv_m, pur_m = prep.process_many(pv_frame, pur_frame) # Transform to InteractionMatrix objects.

```

Important to note is that when you use multiple data inputs as in the example above, you want to process them together, so that all resulting InteractionMatrix objects have the same size. 

### Using an algorithm

RecPack provides a number of state-of-the-art algorithm implementations. 

To get the list of all algorithms see [algorithms documentation page](https://adrem-recommenders.gitlab.io/recpack/recpack.algorithms.html):

To create an algorithm instance use:

```python
import recpack.algorithms
from recpack.algorithms import EASE

algo = EASE(lambda=1000)
```
For information on which parameters can be used for which algorithm see their individual documentation pages.

#### Implementing a new algorithm
Should you want to implement a new algorithm which we do not yet support you should create a subclass of the `Algorithm` base class,
or use one of the more detailed base classes.

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

For a full tutorial on creating your own algorithm check out the [documentation](https://adrem-recommenders.gitlab.io/recpack/guides.create_algorithm.html)
### Selecting a scenario
To evaluate the merits of your algorithms, you use them in specific evaluation scenarios: This is what Scenarios are used for. 
RecPack has implementations for the most commonly used evaluation scenarios:
StrongGeneralization: How well does my algorithm generalize to unseen users? 
Timed: How well does my algorithm predict future interactions for this user?
WeakGeneralization: How well can my algorithm predict a random set of interactions of this user, based on all other interactions?


```python
from recpack.splitters.scenarios import StrongGeneralization
# Construct a splitter object which uses strong generalization to split the data into
# three data objects. Train will contain 50% of the users, validation 20% and test 30%
# The seed parameter is useful for creating reproducible results.
scenario = StrongGeneralizationSplit(0.5, 0.2, seed=42)
scenario.split(pur_m)

scenario.training_data.binary_values  # Training Data, as binary values csr_matrix
validation_data_in, validation_data_out = scenario.validation_data  # Validation Data: Split across historical interactions -> interactions to predict
test_data_in, test_data_out = scenario.test_data  # Test Data: Split across historical interactions -> interactions to predict
```

### Creating the pipeline
When creating the pipeline you will connect all the components and select metrics which the pipeline will compute.

```python
import recpack.pipelines
# Construct a pipeline which computes NDCG and Recall @ 10, 20, 50 and 100
# Only a single algorithm is evaluated, but you could select multiple algorithms to be evaluated at the same time.
pipeline_builder = recpack.pipelines.PipelineBuilder('demo')
pipeline_builder.set_train_data(scenario.training_data)
pipeline_builder.set_test_data(scenario.test_data)

pipeline_builder.add_algorithm('Popularity')
pipeline_builder.add_metric('NormalizedDiscountedCumulativeGainK', [10, 20, 30])
pipeline_builder.add_metric('RecallK', [10, 20, 30])

pipeline = pipeline_builder.build()

pipeline.run()

# Get the metric results.
# This will be a dict with the results of the run.
# Turning it into a dataframe makes reading easier
pd.DataFrame.from_dict(pipeline.get_metrics()) 
```
