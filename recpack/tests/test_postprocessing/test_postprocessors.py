# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
from recpack.matrix import InteractionMatrix
from recpack.postprocessing.postprocessors import Postprocessor
import recpack.postprocessing.filters as filters
import numpy as np
from scipy.sparse import csr_matrix

AMOUNT_OF_USERS = 100
AMOUNT_OF_ITEMS = 25
AMOUNT_SELECTED = 5


@pytest.mark.parametrize(
    "filter_items",
    [(np.random.choice(range(AMOUNT_OF_ITEMS), np.random.randint(AMOUNT_SELECTED), replace=False),)],
)
def test_add_filter(filter_items):
    post_processor = Postprocessor()

    post_processor.add_filter(filters.ExcludeItems(filter_items))
    post_processor.add_filter(filters.SelectItems(filter_items))

    assert type(post_processor.filters[0]) == filters.ExcludeItems
    assert type(post_processor.filters[1]) == filters.SelectItems

    post_processor.add_filter(filters.SelectItems(filter_items), 0)

    assert type(post_processor.filters[0]) == filters.SelectItems
    assert type(post_processor.filters[1]) == filters.ExcludeItems
    assert type(post_processor.filters[2]) == filters.SelectItems

    post_processor.add_filter(filters.ExcludeItems(filter_items), 5)
    # Following list.insert behaviour, specifying an index beyond the maximal index present will append.
    assert type(post_processor.filters[0]) == filters.SelectItems
    assert type(post_processor.filters[1]) == filters.ExcludeItems
    assert type(post_processor.filters[2]) == filters.SelectItems
    assert type(post_processor.filters[3]) == filters.ExcludeItems


@pytest.mark.parametrize(
    "prediction_matrix1, prediction_matrix2, filter_items",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(
                range(AMOUNT_OF_ITEMS),
                np.random.randint(1, AMOUNT_SELECTED),
                replace=False,
            ),
        ),
    ],
)
def test_process_many(prediction_matrix1, prediction_matrix2, filter_items):
    post_processor = Postprocessor()
    post_processor.add_filter(filters.ExcludeItems(filter_items))
    filter_preds = post_processor.process_many(prediction_matrix1, prediction_matrix2)

    assert filter_preds[0].shape == prediction_matrix1.shape
    assert filter_preds[1].shape == prediction_matrix2.shape

    assert not filter_preds[0][:, filter_items].toarray().any()
    assert not filter_preds[1][:, filter_items].toarray().any()


@pytest.mark.parametrize(
    "prediction_matrix1, prediction_matrix2, filter_items",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            csr_matrix(np.random.random_sample(size=(2 * AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS), np.random.randint(1, AMOUNT_SELECTED), replace=False),
        ),
    ],
)
def test_process_many_diff_size(prediction_matrix1, prediction_matrix2, filter_items):
    post_processor = Postprocessor()
    post_processor.add_filter(filters.ExcludeItems(filter_items))
    with pytest.raises(ValueError):
        post_processor.process_many(prediction_matrix1, prediction_matrix2)


@pytest.mark.parametrize(
    "prediction_matrix, filter_items1, filter_items2",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(
                range(AMOUNT_OF_ITEMS),
                np.random.randint(1, AMOUNT_SELECTED),
                replace=False,
            ),
            np.random.choice(
                range(AMOUNT_OF_ITEMS),
                np.random.randint(1, AMOUNT_SELECTED),
                replace=False,
            ),
        ),
    ],
)
def test_process(prediction_matrix, filter_items1, filter_items2):
    post_processor = Postprocessor()
    post_processor.add_filter(filters.ExcludeItems(filter_items1))
    post_processor.add_filter(filters.SelectItems(filter_items2))

    filter_pred = post_processor.process(prediction_matrix)
    excluded = [z for z in range(filter_pred.shape[1]) if z not in filter_items2]

    assert filter_pred.shape == prediction_matrix.shape

    assert not filter_pred[:, filter_items1].toarray().any()
    assert not filter_pred[:, excluded].toarray().any()
