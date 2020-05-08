import recpack
from recpack.data_matrix import DataM
import recpack.metrics
import recpack.pipelines
import recpack.algorithms
import recpack.splitters.scenarios as scenarios
import pandas as pd
import pytest


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {'userId': [1, 1, 1, 0, 0, 0], 'movieId': [1, 3, 4, 0, 2, 4], 'timestamp': [15, 26, 29, 10, 22, 34]}

    df = pd.DataFrame.from_dict(input_dict)
    data = DataM.create_from_dataframe(df, 'movieId', 'userId', 'timestamp')
    return data


def test_pipeline():
    data = generate_data()
    scenario = scenarios.TrainingInTestOutTimed(20)
    algo = recpack.algorithms.algorithm_registry.get('popularity')(K=2)

    p = recpack.pipelines.Pipeline([algo], ['NDCG', 'Recall'], [2])
    p.run(*scenario.split(data))

    metrics = p.get()
    assert algo.name in metrics
    assert "NDCG_K_2" in metrics[algo.name]
    assert "Recall_K_2" in metrics[algo.name]