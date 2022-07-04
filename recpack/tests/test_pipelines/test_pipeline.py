from unittest.mock import MagicMock, patch

from recpack.postprocessing.filters import PostFilter


class MockFilter(PostFilter):
    def __init__(self):
        self.apply_calls = 0

    def apply(self, X):
        self.apply_calls += 1
        return X


def test_pipeline(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    metrics = pipeline.get_metrics()
    assert len(metrics) == len(pipeline.algorithm_entries)

    assert len(metrics[list(metrics.keys())[0]]) == len(pipeline.metric_entries)


def test_pipeline_save_metrics(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    mocker = MagicMock()
    with patch("recpack.pipelines.pipeline.pd.DataFrame.to_json", mocker):
        pipeline.save_metrics()

        mocker.assert_called_once_with(f"{pipeline_builder.results_directory}/results.json")


def test_pipeline_optimisation_results(pipeline_builder_optimisation):
    pipeline = pipeline_builder_optimisation.build()

    pipeline.run()

    metrics = pipeline.get_metrics()
    assert len(metrics) == len(pipeline.algorithm_entries)

    assert len(metrics[list(metrics.keys())[0]]) == len(pipeline.metric_entries)

    # 3 parameters tried for knn and 2 for ease.
    print(pipeline.optimisation_results)
    assert pipeline.optimisation_results.shape[0] == 5


def test_pipeline_with_filters_applied(pipeline_builder):

    mock_filter = MockFilter()

    pipeline_builder.add_post_filter(mock_filter)
    pipe = pipeline_builder.build()
    pipe.run()
    assert mock_filter.apply_calls == 2


def test_pipeline_with_filters_applied_optimisation(pipeline_builder_optimisation):

    mock_filter = MockFilter()

    pipeline_builder_optimisation.add_post_filter(mock_filter)
    pipe = pipeline_builder_optimisation.build()
    pipe.run()
    assert mock_filter.apply_calls == 7  # 3 ItemKNN, 2 EASE, 2 final eval
