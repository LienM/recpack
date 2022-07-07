# RecPack
RecPack is an experimentation toolkit for top-N recommendation using implicit feedback data written in Python, with a familiar interface and clear documentation. Its goals is to support researchers who advance the state-of-the-art in top-N recommendation to write reproducible and reusable experiments. RecPack comes with a range of different datasets, recommendation scenarios, state-of-the-art baselines and metrics. Wherever possible, RecPack sets sensible defaults. For example, hyperparameters of all recommendation algorithms included in RecPack are initialized to the best performing settings found in the original experiments. The design of RecPack is heavily inspired by the interface of scikit-learn, a popular Python package for classification, regression and clustering tasks. Data scientists who are familiar with scikit-learn will already have an intuitive understanding of how to work with RecPack. On top of this, RecPack was developed with a production mindset: All contributions are rigorously reviewed and tested. The RecPack maintainers strive to maintain a test coverage of more than ninety percent at all times.


## Installation

All released versions of recpack are published on [Pypi](https://pypi.org/project/recpack/), and can be installed using:

`pip install recpack`

## documentation
Documentation and tutorials can be found at https://recpack.froomle.ai

## Usage
RecPack provides a framework for experimentation with recommendation algorithms. 
It comes pre-packed with a number of commonly used evaluation scenarios (scenarios),
evaluation metrics (metrics) and state-of-the-art algorithm implementations (algorithms).
New algorithms and evaluation scenarios can be added easily, by subclassing the appropriate base classes. 
A number of lower level data splitters are provided that can be used to build up more complex evaluation scenarios.

Users can choose between the Pipeline interface, and manually connecting components for running experiments. 
Pipelines are easier to set up comparisons between algorithms, for full control manual linking is available.

For details and examples, check out the [quickstart documentation](recpack.froomle.ai/guides.quickstart.html)

### Selecting a scenario
To evaluate the merits of your algorithms, you use them in specific evaluation scenarios: This is what Scenarios are used for. 
RecPack has implementations for the most commonly used evaluation scenarios:

* StrongGeneralization: How well does my algorithm generalize to unseen users? 
* Timed: How well does my algorithm predict future interactions for this user?
* WeakGeneralization: How well can my algorithm predict a random set of interactions of this user, based on all other interactions?


```python
from recpack.scenarios import StrongGeneralization
scenario = StrongGeneralizationSplit(0.5, 0.2, seed=42)
scenario.split(pur_m)

scenario.full_training_data.binary_values  # full training Data, as binary values csr_matrix
validation_training_data.binary_values # Training data to use when optimising parameters
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
pipeline_builder.set_data_from_scenario(scenario)

pipeline_builder.add_algorithm('Popularity')
pipeline_builder.add_metric('NormalizedDiscountedCumulativeGainK', [10, 20, 50, 100])
pipeline_builder.add_metric('RecallK', [10, 20, 30, 100])

pipeline = pipeline_builder.build()

pipeline.run()

# Get the metric results.
pipeline.get_metrics()
```
