import os
import pytest
import time

from recpack.pipelines import PipelineBuilder
from recpack.postprocessing.filters import ExcludeItems
from recpack.scenarios import Timed


def test_pipeline_builder(mat):
    pb = PipelineBuilder()
    assert pb.base_path == os.getcwd()

    # Build empty pipeline
    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No metrics specified, can't construct pipeline"

    # Add 1 or multiple metrics
    pb.add_metric("CalibratedRecallK", 20)
    assert len(pb.metric_entries) == 1
    pb.add_metric("CalibratedRecallK", [10, 30])
    assert len(pb.metric_entries) == 3

    # Build pipeline without algorithms
    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "No algorithms specified, can't construct pipeline"

    # Add algorithms
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.add_algorithm("ItemKNN", grid={"K": [10, 20, 50, 100]})
    pb.add_algorithm("Prod2Vec", grid={"num_components": [10, 20]}, params={"learning_rate": 0.1})
    assert len(pb.algorithm_entries) == 3
    assert pb.algorithm_entries[0].grid == {}
    assert pb.algorithm_entries[0].params == {"K": 20}
    assert pb.algorithm_entries[1].params == {}
    assert pb.algorithm_entries[2].grid == {"num_components": [10, 20]}
    assert pb.algorithm_entries[2].params == {"learning_rate": 0.1}

    # Build pipeline that needs to be optimized without optimization metric
    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == (
        "No optimisation metric selected to perform "
        "requested hyperparameter optimisation, "
        "can't construct pipeline."
    )

    pb.set_optimisation_metric("CalibratedRecallK", 20)

    # Build pipeline without training data
    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert error.value.args[0] == "No full training data available, can't construct pipeline."

    pb.set_full_training_data(mat)
    assert pb.full_training_data.shape == mat.shape

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
        error.value.args[0] == "No validation data available to perform "
        "the requested hyperparameter optimisation, "
        "can't construct pipeline."
    )

    pb.set_validation_data((mat, mat))
    assert len(pb.validation_data) == 2
    assert pb.validation_data[0].shape == mat.shape

    # Build pipeline fails because there is no validation_training data
    with pytest.raises(RuntimeError) as error:
        pb.build()

    assert (
        error.value.args[0] == "No validation training data available to perform "
        "the requested hyperparameter optimisation, "
        "can't construct pipeline."
    )

    pb.set_validation_training_data(mat)
    pipeline = pb.build()

    assert len(pipeline.metric_entries) == 3
    assert len(pipeline.algorithm_entries) == 3
    assert pipeline.optimisation_metric_entry.name == "CalibratedRecallK"
    assert pipeline.optimisation_metric_entry.K == 20
    assert pipeline.validation_training_data.shape == mat.shape
    assert pipeline.test_data_in is not None
    assert pipeline.test_data_out is not None
    assert pipeline.validation_data is not None


def test_pipeline_builder_no_validation_torch(pipeline_builder):
    pipeline_builder.add_algorithm("BPRMF", params={"learning_rate": 0.1})

    with pytest.raises(RuntimeError) as error:
        pipeline_builder.build()

    assert error.value.args[0] == (
        "No validation data available to perform "
        "the requested hyperparameter optimisation, "
        "can't construct pipeline."
    )


def test_pipeline_builder_no_optimisation(mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_full_training_data(mat)
    pb.set_test_data((mat, mat))

    pipeline = pb.build()

    assert len(pipeline.metric_entries) == 1


def test_pipeline_duplicate_metric(mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_full_training_data(mat)
    pb.set_test_data((mat, mat))

    assert len(pb.metric_entries) == 1
    pipeline = pb.build()

    assert len(pipeline.metric_entries) == 1


def test_pipeline_mismatching_shapes_test(mat, larger_mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_full_training_data(mat)
    pb.set_test_data((larger_mat, larger_mat))

    with pytest.raises(RuntimeError) as error:
        pb.build()
    assert error.value.args[0] == "Shape mismatch between test and training data"


def test_pipeline_mismatching_shapes_validation(mat, larger_mat):
    pb = PipelineBuilder()
    pb.add_metric("CalibratedRecallK", 20)
    pb.add_algorithm("ItemKNN", params={"K": 20})
    pb.set_validation_training_data(mat)
    pb.set_full_training_data(mat)
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

    assert error.value.args[0] == "Incorrect value, expected tuple with data_in and data_out"


def test_pipeline_builder_bad_validation_data(mat):
    pb = PipelineBuilder()

    with pytest.raises(TypeError) as error:
        pb.set_validation_data(None)
    with pytest.raises(TypeError) as error:
        pb.set_validation_data(mat)
    with pytest.raises(ValueError) as error:
        pb.set_validation_data((mat,))

    assert error.value.args[0] == "Incorrect value, expected tuple with data_in and data_out"


def test_default_name():
    pb = PipelineBuilder()
    time.sleep(0.1)
    pb2 = PipelineBuilder()

    assert pb.folder_name != pb2.folder_name


def test_add_post_filter(pipeline_builder):
    pipeline_builder.add_post_filter(ExcludeItems)

    pipe = pipeline_builder.build()
    assert len(pipe.post_processor.filters) == 1


def test_remove_history(pipeline_builder):
    assert pipeline_builder.build().remove_history

    pipeline_builder.remove_history = False
    assert not pipeline_builder.build().remove_history


def test_set_data_from_scenario(mat):
    pb = PipelineBuilder()

    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 3)
    pb.add_algorithm("ItemKNN", params={"K": 2})
    pb.add_algorithm("EASE", params={"l2": 2})

    scenario = Timed(t=3, t_validation=2, validation=True)
    scenario.split(mat)
    pb.set_data_from_scenario(scenario)

    assert len(pb.test_data) == 2
    assert len(pb.validation_data) == 2
    assert pb.full_training_data.shape == mat.shape
    assert pb.validation_training_data.shape == mat.shape
