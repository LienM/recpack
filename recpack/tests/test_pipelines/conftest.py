# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest

from recpack.pipelines import PipelineBuilder, GridSearchInfo


@pytest.fixture()
def pipeline_builder(mat):
    name = "test_builder"

    pb = PipelineBuilder(name)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 3)
    pb.add_algorithm("ItemKNN", params={"K": 2})
    pb.add_algorithm("EASE", params={"l2": 2})
    pb.set_full_training_data(mat)
    pb.set_test_data((mat, mat))

    return pb


@pytest.fixture()
def pipeline_builder_optimisation(mat):
    name = "test_builder"

    pb = PipelineBuilder(name)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 3)
    pb.add_algorithm("ItemKNN", optimisation_info=GridSearchInfo({"K": [1, 2, 3]}))
    pb.add_algorithm("EASE", optimisation_info=GridSearchInfo({"l2": [2, 10]}))
    pb.set_full_training_data(mat)
    pb.set_validation_training_data(mat)
    pb.set_optimisation_metric("CalibratedRecallK", 2)
    pb.set_test_data((mat, mat))
    pb.set_validation_data((mat, mat))

    return pb


@pytest.fixture()
def pipeline_builder_optimisation_no_algos(larger_mat):
    name = "test_builder"

    pb = PipelineBuilder(name)
    pb.add_metric("CalibratedRecallK", 2)
    pb.add_metric("CalibratedRecallK", 3)
    pb.set_full_training_data(larger_mat)
    pb.set_validation_training_data(larger_mat)
    pb.set_optimisation_metric("CalibratedRecallK", 2)
    pb.set_test_data((larger_mat, larger_mat))
    pb.set_validation_data((larger_mat, larger_mat))

    return pb
