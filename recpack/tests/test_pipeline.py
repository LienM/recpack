import os
import pandas as pd
import pytest
import time
from unittest.mock import MagicMock, mock_open, patch
import yaml

import recpack
from recpack.data.matrix import InteractionMatrix
import recpack.metrics
import recpack.pipeline
import recpack.algorithms
import recpack.splitters.scenarios as scenarios


@pytest.fixture()
def pipeline_builder(mat):
    name = "test_builder"

    pb = recpack.pipeline.PipelineBuilder(name)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_algorithm("ItemKNN", params={"K": 2})
    pb.add_algorithm("EASE", params={"l2": 2})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))

    return pb


def test_pipeline_builder(mat):
    pb = recpack.pipeline.PipelineBuilder()
    assert pb.path == os.getcwd()
    assert pb.metrics == []
    assert pb.algorithms == []

    assert pb.test_data is None
    assert pb.train_data is None
    assert pb.validation_data is None

    assert pb.optimisation_metric is None

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No metrics specified, can't construct pipeline"

    pb.add_metric("CalibratedRecallK", 20)
    assert len(pb.metrics) == 1
    pb.add_metric("CalibratedRecallK", [10, 30])
    assert len(pb.metrics) == 3

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No algorithms specified, can't construct pipeline"

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

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No optimisation metric selected"

    pb.set_optimisation_metric("CalibratedRecallK", 20)

    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert (
        error.value.args[0] == "No training data available, can't construct pipeline."
    )

    pb.set_train_data(mat)
    assert pb.train_data.shape == mat.shape

    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert error.value.args[0] == "No test data available, can't construct pipeline."

    pb.set_test_data((mat, mat))
    assert len(pb.test_data) == 2
    assert pb.test_data[0].shape == mat.shape

    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert (
        error.value.args[0] == "No validation data available, can't construct pipeline."
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


def test_pipeline_builder_no_optimisation(mat):
    pb = recpack.pipeline.PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))

    pipeline = pb.build()

    assert len(pipeline.metrics) == 1
    assert pipeline.optimisation_metric is None
    assert pipeline.validation_data is None


def test_pipeline_duplicate_metric(mat):
    pb = recpack.pipeline.PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))

    assert len(pb.metrics) == 3
    pipeline = pb.build()

    assert len(pipeline.metrics) == 1


def test_pipeline_mismatching_shapes_test(mat, larger_mat):
    pb = recpack.pipeline.PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((larger_mat, larger_mat))

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "Shape mismatch between test and training data"


def test_pipeline_mismatching_shapes_validation(mat, larger_mat):
    pb = recpack.pipeline.PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_train_data(mat)
    pb.set_test_data((mat, mat))
    pb.set_validation_data((larger_mat, larger_mat))

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "Shape mismatch between validation and training data"


def test_pipeline_builder_bad_test_data(mat):
    pb = recpack.pipeline.PipelineBuilder()

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
    pb = recpack.pipeline.PipelineBuilder()

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


def test_pipeline_builder_construct_config_dict(pipeline_builder):

    d = pipeline_builder._construct_config_dict()

    assert len(d["metrics"]) == 2
    assert d["metrics"][0]["name"] == "CalibratedRecallK"
    assert d["metrics"][0]["K"] == 2

    assert len(d["algorithms"]) == 2
    assert d["algorithms"][0]["name"] == "ItemKNN"
    assert d["algorithms"][1]["name"] == "EASE"

    assert (
        d["train_data"]["filename"]
        == f"{pipeline_builder.path}/{pipeline_builder.name}/train"
    )
    assert (
        d["test_data"]["data_in"]["filename"]
        == f"{pipeline_builder.path}/{pipeline_builder.name}/test_in"
    )
    assert (
        d["test_data"]["data_out"]["filename"]
        == f"{pipeline_builder.path}/{pipeline_builder.name}/test_out"
    )

    assert d["validation_data"] is None


def test_save(pipeline_builder, mat):

    mocker = mock_open()
    mocker2 = mock_open()
    mocker3 = MagicMock()
    with patch("recpack.data.matrix.pd.DataFrame.to_csv", mocker3):
        with patch("recpack.data.matrix.open", mocker2):
            with patch("recpack.pipeline.open", mocker):
                pipeline_builder.save()

    assert mocker2.call_count == 3

    assert mocker2.call_args_list[0].args == (
        f"{pipeline_builder.path}/{pipeline_builder.name}/train_metadata.yaml",
        "w",
    )
    assert mocker2.call_args_list[1].args == (
        f"{pipeline_builder.path}/{pipeline_builder.name}/test_in_metadata.yaml",
        "w",
    )
    assert mocker2.call_args_list[2].args == (
        f"{pipeline_builder.path}/{pipeline_builder.name}/test_out_metadata.yaml",
        "w",
    )

    assert mocker.call_count == 1

    mocker.assert_called_with(
        f"{pipeline_builder.path}/{pipeline_builder.name}/config.yaml", "w"
    )
    handler = mocker()
    handler.write.assert_called_with(
        yaml.safe_dump(pipeline_builder._construct_config_dict())
    )


def test_load(pipeline_builder, mat):

    mocker = mock_open(
        read_data=yaml.safe_dump(pipeline_builder._construct_config_dict())
    )
    mocker2 = mock_open(read_data=yaml.safe_dump(mat.metadata.to_dict()))
    mocker3 = MagicMock(return_value=mat._df)

    pb2 = recpack.pipeline.PipelineBuilder(
        name=pipeline_builder.name, path=pipeline_builder.path
    )

    with patch("recpack.data.matrix.pd.read_csv", mocker3):
        with patch("recpack.data.matrix.open", mocker2):
            with patch("recpack.pipeline.open", mocker):

                pb2.load()

    assert pb2.metrics == pipeline_builder.metrics
    assert pb2.algorithms == pipeline_builder.algorithms


def test_default_name():
    pb = recpack.pipeline.PipelineBuilder()
    time.sleep(0.1)
    pb2 = recpack.pipeline.PipelineBuilder()

    assert pb.name != pb2.name


def test_pipeline(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    metrics = pipeline.get()
    assert len(metrics) == len(pipeline.algorithms)

    assert len(metrics[list(metrics.keys())[0]]) == len(pipeline.metrics)
