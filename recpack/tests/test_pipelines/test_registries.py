import pytest

from recpack.pipelines.registries import ALGORITHM_REGISTRY, METRIC_REGISTRY


def test_metric_registry():
    assert "CalibratedRecallK" in METRIC_REGISTRY
    assert "HitK" in METRIC_REGISTRY
    assert "NDCGK" in METRIC_REGISTRY


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
