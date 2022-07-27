import pytest
import os


@pytest.fixture()
def dataset_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")
