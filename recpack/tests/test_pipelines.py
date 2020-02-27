import recpack
import recpack.helpers
import recpack.splits
import recpack.evaluate
import recpack.pipelines
import recpack.algorithms
import pandas as pd


def generate_data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {'userId': [1, 1, 1, 0, 0, 0], 'movieId': [1, 3, 4, 0, 2, 4], 'timestamp': [15, 26, 29, 10, 22, 34]}

    df = pd.DataFrame.from_dict(input_dict)
    data = recpack.helpers.create_data_M_from_pandas_df(df, 'movieId', 'userId', 'timestamp')
    return data


def test_pipeline():
    data = generate_data()
    splitter = recpack.splits.TimedSplit(20, None)
    evaluator = recpack.evaluate.TrainingInTestOutEvaluator()
    algo = recpack.algorithms.get_algorithm('popularity')(K=2)

    p = recpack.pipelines.Pipeline(splitter, [algo], evaluator, ['NDCG', 'Recall'], [2])
    p.run(data)

    metrics = p.get()
    assert algo.name in metrics
    assert "NDCG_K_2" in metrics[algo.name]
    assert "Recall_K_2" in metrics[algo.name]


def test_parameter_generator_pipeline():
    data = generate_data()
    NUM_SLICES = 3
    splitter = recpack.splits.TimedSplit
    evaluator = recpack.evaluate.TrainingInTestOutEvaluator
    algo = recpack.algorithms.get_algorithm('popularity')(K=2)
    parameter_generator = recpack.pipelines.TemporalSWParameterGenerator(10, None, 10, NUM_SLICES)
    p = recpack.pipelines.ParameterGeneratorPipeline(
        parameter_generator, splitter, [algo], evaluator, ['NDCG', 'Recall'], [2]
    )

    p.run(data)

    metrics = p.get()

    assert len(metrics) == NUM_SLICES
    for metric in metrics:
        assert algo.name in metric
        assert "NDCG_K_2" in metric[algo.name]
        assert "Recall_K_2" in metric[algo.name]
