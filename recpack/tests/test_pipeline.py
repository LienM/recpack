import recpack
from recpack.data.matrix import InteractionMatrix
import recpack.metrics
import recpack.pipeline
import recpack.algorithms
import recpack.splitters.scenarios as scenarios
import pandas as pd


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {
        InteractionMatrix.USER_IX: [1, 1, 1, 0, 0, 0],
        InteractionMatrix.ITEM_IX: [1, 3, 4, 0, 2, 4],
        InteractionMatrix.TIMESTAMP_IX: [15, 26, 29, 10, 22, 34],
    }

    df = pd.DataFrame.from_dict(input_dict)
    data = InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    return data


# TODO Add tests for the if-else branches in pipeline now


def test_pipeline():
    data = generate_data()
    scenario = scenarios.Timed(20)
    scenario.split(data)

    algo = recpack.algorithms.Popularity(K=2)

    p = recpack.pipeline.Pipeline(
        [algo], ["NormalizedDiscountedCumulativeGainK", "RecallK"], [2]
    )

    p.run(scenario.training_data, scenario.test_data)

    metrics = p.get()
    assert algo.identifier in metrics
    assert "NormalizedDiscountedCumulativeGainK_K_2" in metrics[algo.identifier]
    assert "RecallK_K_2" in metrics[algo.identifier]
