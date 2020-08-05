import recpack
from recpack.data.data_matrix import DataM
import recpack.metrics
import recpack.pipeline
import recpack.algorithms
import recpack.splitters.scenarios as scenarios
import pandas as pd


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {
        "userId": [1, 1, 1, 0, 0, 0],
        "movieId": [1, 3, 4, 0, 2, 4],
        "timestamp": [15, 26, 29, 10, 22, 34],
    }

    df = pd.DataFrame.from_dict(input_dict)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", timestamp_ix="timestamp"
    )
    return data


# TODO Add tests for the if-else branches in pipeline now


def test_pipeline():
    data = generate_data()
    scenario = scenarios.TrainingInTestOutTimed(20)
    scenario.split(data)

    algo = recpack.algorithms.algorithm_registry.get("popularity")(K=2)

    p = recpack.pipeline.Pipeline([algo], ["NDCG", "Recall"], [2])

    p.run(scenario.training_data, scenario.test_data)

    metrics = p.get()
    assert algo.identifier in metrics
    assert "NDCG_K_2" in metrics[algo.identifier]
    assert "Recall_K_2" in metrics[algo.identifier]
