import os
import pandas as pd
import pytest
import time
from unittest.mock import MagicMock, mock_open, patch
import yaml

from recpack.pipelines import PipelineBuilder, ALGORITHM_REGISTRY, METRIC_REGISTRY


# ---- TEST REGISTRIES
def test_metric_registry():
    assert "CalibratedRecallK" in METRIC_REGISTRY
    assert "HitK" in METRIC_REGISTRY
    assert "NormalizedDiscountedCumulativeGainK" in METRIC_REGISTRY


def test_algorithm_registry():
    assert "ItemKNN" in ALGORITHM_REGISTRY
    assert "MultVAE" in ALGORITHM_REGISTRY
    assert "NMF" in ALGORITHM_REGISTRY


def test_adding_key():
    class NewAlgorithm:
        def __init__(self):
            self.hello = "World"

    ALGORITHM_REGISTRY.register("hello", NewAlgorithm)
    assert "hello" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["hello"]().hello == "World"

    # Don't allow duplicate
    with pytest.raises(KeyError):
        ALGORITHM_REGISTRY.register("hello", NewAlgorithm)


# ---- TEST PIPELINE
@pytest.fixture()
def pipeline_builder(mat):
    name = "test_builder"

    pb = PipelineBuilder(name)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 3)
    pb.add_algorithm("ItemKNN", params={"K": 2})
    pb.add_algorithm("EASE", params={"l2": 2})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))

    return pb


@pytest.fixture()
def pipeline_builder_optimisation(mat):
    name = "test_builder"

    pb = PipelineBuilder(name)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 3)
    pb.add_algorithm("ItemKNN", grid={"K": [1, 2, 3]})
    pb.add_algorithm("EASE", grid={"l2": [2, 10]})
    pb.set_train_data(mat)
    pb.set_optimisation_metric("CalibratedRecallK", 2)
    pb.set_test_data((mat, mat))
    pb.set_validation_data((mat, mat))

    return pb


def test_pipeline_builder(mat):
    pb = PipelineBuilder()
    assert pb.base_path == os.getcwd()

    # Build empty pipeline
    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No metrics specified, can't construct pipeline"

    # Add 1 or multiple metrics
    pb.add_metric("CalibratedRecallK", 20)
    assert len(pb.metrics) == 1
    pb.add_metric("CalibratedRecallK", [10, 30])
    assert len(pb.metrics) == 3

    # Build pipeline without algorithms
    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No algorithms specified, can't construct pipeline"

    # Add algorithms
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.add_algorithm("ItemKNN", grid={"K": [10, 20, 50, 100]})
    pb.add_algorithm(
        "Prod2Vec", grid={"embedding_size": [10, 20]}, params={"learning_rate": 0.1}
    )
    assert len(pb.algorithms) == 3
    assert pb.algorithms[0].grid == {}
    assert pb.algorithms[0].params == {"K": 20}
    assert pb.algorithms[1].params == {}
    assert pb.algorithms[2].grid == {"embedding_size": [10, 20]}
    assert pb.algorithms[2].params == {"learning_rate": 0.1}

    # Build pipeline that needs to be optimized without optimization metric
    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == (
        "No optimisation metric selected to perform requested hyperparameter optimisation,"
        "can't construct pipeline."
    )

    pb.set_optimisation_metric("CalibratedRecallK", 20)

    # Build pipeline without training data
    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert (
        error.value.args[0] == "No training data available, can't construct pipeline."
    )

    pb.set_train_data(mat)
    assert pb.train_data.shape == mat.shape

    # Build pipeline without test data
    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert error.value.args[0] == "No test data available, can't construct pipeline."

    pb.set_test_data((mat, mat))
    assert len(pb.test_data) == 2
    assert pb.test_data[0].shape == mat.shape

    # Build pipeline that needs to be optimized without validation data
    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert (
        error.value.args[0]
        == "No validation data available to perform the requested hyperparameter optimisation"
        ", can't construct pipeline."
    )

    pb.set_validation_data((mat, mat))
    assert len(pb.validation_data) == 2
    assert pb.validation_data[0].shape == mat.shape

    pipeline = pb.build()

    assert len(pipeline.metrics) == 3
    assert len(pipeline.algorithms) == 3
    assert pipeline.optimisation_metric.name == "CalibratedRecallK"
    assert pipeline.optimisation_metric.K == 20
    assert pipeline.train_data.shape == mat.shape
    assert pipeline.test_data is not None
    assert pipeline.validation_data is not None


def test_pipeline_builder_no_validation_torch(pipeline_builder):
    pipeline_builder.add_algorithm("BPRMF", params={"learning_rate": 0.1})

    with pytest.raises(RuntimeError) as error:
        pipeline_builder.build()

    assert error.value.args[0] == (
        "No validation data available to perform the requested hyperparameter optimisation"
        ", can't construct pipeline."
    )


def test_pipeline_builder_no_optimisation(mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))

    pipeline = pb.build()

    assert len(pipeline.metrics) == 1


def test_pipeline_duplicate_metric(mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))

    assert len(pb.metrics) == 1
    pipeline = pb.build()

    assert len(pipeline.metrics) == 1


def test_pipeline_mismatching_shapes_test(mat, larger_mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((larger_mat, larger_mat))

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "Shape mismatch between test and training data"


def test_pipeline_mismatching_shapes_validation(mat, larger_mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))
    pb.set_validation_data((larger_mat, larger_mat))

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "Shape mismatch between validation and training data"


def test_pipeline_builder_bad_test_data(mat):
    pb = PipelineBuilder()

    with pytest.raises(TypeError) as error:
        pb.set_test_data(None)
    with pytest.raises(TypeError) as error:
        pb.set_test_data(mat)
    with pytest.raises(ValueError) as error:
        pb.set_test_data((mat,))

    assert (
        error.value.args[0]
        == "Incorrect value, expected tuple with data_in and data_out"
    )


def test_pipeline_builder_bad_validation_data(mat):
    pb = PipelineBuilder()

    with pytest.raises(TypeError) as error:
        pb.set_validation_data(None)
    with pytest.raises(TypeError) as error:
        pb.set_validation_data(mat)
    with pytest.raises(ValueError) as error:
        pb.set_validation_data((mat,))

    assert (
        error.value.args[0]
        == "Incorrect value, expected tuple with data_in and data_out"
    )


def test_pipeline_builder_pipeline_config(pipeline_builder):

    d = pipeline_builder._pipeline_config

    assert len(d["metrics"]) == 2
    assert d["metrics"][0]["name"] == "CalibratedRecallK"
    assert d["metrics"][0]["K"] == 2

    assert len(d["algorithms"]) == 2
    assert d["algorithms"][0]["name"] == "ItemKNN"
    assert d["algorithms"][1]["name"] == "EASE"


def test_save(pipeline_builder, mat):

    mocker = mock_open()
    mocker2 = mock_open()
    mocker3 = MagicMock()
    with patch("recpack.data.matrix.pd.DataFrame.to_csv", mocker3):
        with patch("recpack.data.matrix.open", mocker2):
            with patch("recpack.pipelines.pipeline.open", mocker):
                pipeline_builder.save()

    assert mocker2.call_count == 3

    assert mocker2.call_args_list[0].args == (
        f"{pipeline_builder.base_path}/{pipeline_builder.folder_name}/train_properties.yaml",
        "w",
    )
    assert mocker2.call_args_list[1].args == (
        f"{pipeline_builder.base_path}/{pipeline_builder.folder_name}/test_in_properties.yaml",
        "w",
    )
    assert mocker2.call_args_list[2].args == (
        f"{pipeline_builder.base_path}/{pipeline_builder.folder_name}/test_out_properties.yaml",
        "w",
    )

    mocker.assert_called_with(
        f"{pipeline_builder.base_path}/{pipeline_builder.folder_name}/config.yaml", "w"
    )
    handler = mocker()
    handler.write.assert_called_with(yaml.safe_dump(pipeline_builder._pipeline_config))


def test_load(pipeline_builder, mat):

    mocker = mock_open(read_data=yaml.safe_dump(pipeline_builder._pipeline_config))
    mocker2 = mock_open(read_data=yaml.safe_dump(mat.properties.to_dict()))
    mocker3 = MagicMock(return_value=mat._df)

    pb2 = PipelineBuilder(
        folder_name=pipeline_builder.folder_name, base_path=pipeline_builder.base_path
    )

    with patch("recpack.data.matrix.pd.read_csv", mocker3):
        with patch("recpack.data.matrix.open", mocker2):
            with patch("recpack.pipelines.pipeline.open", mocker):

                pb2.load()

    assert pb2.metrics == pipeline_builder.metrics
    assert pb2.algorithms == pipeline_builder.algorithms


def test_default_name():
    pb = PipelineBuilder()
    time.sleep(0.1)
    pb2 = PipelineBuilder()

    assert pb.folder_name != pb2.folder_name


def test_pipeline(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    metrics = pipeline.get_metrics()
    assert len(metrics) == len(pipeline.algorithms)

    assert len(metrics[list(metrics.keys())[0]]) == len(pipeline.metrics)


def test_pipeline_save_metrics(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    mocker = MagicMock()
    with patch("recpack.pipelines.pipeline.pd.DataFrame.to_json", mocker):
        pipeline.save_metrics()

        mocker.assert_called_once_with(
            f"{pipeline_builder.results_directory}/results.json"
        )


def test_pipeline_optimisation_results(pipeline_builder_optimisation):
    pipeline = pipeline_builder_optimisation.build()

    pipeline.run()

    metrics = pipeline.get_metrics()
    assert len(metrics) == len(pipeline.algorithms)

    assert len(metrics[list(metrics.keys())[0]]) == len(pipeline.metrics)

    # 3 parameters tried for knn and 2 for ease.
    print(pipeline.optimisation_results)
    assert pipeline.optimisation_results.shape[0] == 5
