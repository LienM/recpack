import pytest
from recpack.algorithms.item_metadata_algorithms.jaccard_distance import JaccardDictance


def test_jaccard_distance(metadata_tags_matrix):
    m = JaccardDictance()
    m.fit(metadata_tags_matrix)

    assert m._distances[0, 0] == 0
    assert m._distances[0, 1] == 0.5
    assert m._distances[1, 0] == m._distances[0, 1]

    assert m._distances[0, 3] == 1
    