# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import random
import numpy as np
from scipy.sparse import csr_matrix

import recpack.postprocessing.filters as filters

AMOUNT_OF_USERS = 100
AMOUNT_OF_ITEMS = 25
AMOUNT_SELECTED = 5


@pytest.mark.parametrize(
    "prediction_matrix, items",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS), np.random.randint(1, AMOUNT_SELECTED), replace=False),
        ),
    ],
)
def test_select_items(prediction_matrix, items):
    myfilter = filters.SelectItems(items)
    filter_pred = myfilter.apply(prediction_matrix)

    excluded = [z for z in range(filter_pred.shape[1]) if z not in items]

    assert filter_pred.shape == prediction_matrix.shape
    assert not filter_pred[:, excluded].toarray().any()


@pytest.mark.parametrize(
    "prediction_matrix, items",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS), np.random.randint(1, AMOUNT_SELECTED), replace=False),
        ),
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            random.sample(range(AMOUNT_OF_ITEMS), np.random.randint(1, AMOUNT_SELECTED)),
        ),
    ],
)
def test_select_items_array_like(prediction_matrix, items):
    myfilter = filters.SelectItems(items)
    filter_pred = myfilter.apply(prediction_matrix)

    excluded = [z for z in range(filter_pred.shape[1]) if z not in items]

    assert filter_pred.shape == prediction_matrix.shape
    assert not filter_pred[:, excluded].toarray().any()


@pytest.mark.parametrize(
    "prediction_matrix, items",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(
                range(AMOUNT_OF_ITEMS),
                np.random.randint(1, AMOUNT_SELECTED),
                replace=False,
            ),
        ),
    ],
)
def test_exclude_items(prediction_matrix, items):
    myfilter = filters.ExcludeItems(items)
    filter_pred = myfilter.apply(prediction_matrix)

    assert filter_pred.shape == prediction_matrix.shape
    assert not filter_pred[:, items].toarray().any()


@pytest.mark.parametrize(
    "prediction_matrix1, prediction_matrix2, items",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            csr_matrix(np.random.random_sample(size=(2 * AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(
                range(AMOUNT_OF_ITEMS),
                np.random.randint(1, AMOUNT_SELECTED),
                replace=False,
            ),
        ),
    ],
)
def test_filter_items_multiple(prediction_matrix1, prediction_matrix2, items):
    myfilter = filters.SelectItems(items)
    with pytest.raises(ValueError):
        myfilter.apply_all(prediction_matrix1, prediction_matrix2)


@pytest.mark.parametrize(
    "items",
    [
        (np.random.choice(range(AMOUNT_OF_ITEMS), np.random.randint(1, AMOUNT_SELECTED), replace=False),),
    ],
)
def test_post_filter_item_empty(items):
    myfilter = filters.SelectItems(items)
    assert [] == myfilter.apply_all()


@pytest.mark.parametrize(
    "prediction_matrix, items, filter_class",
    [
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS), 0, replace=False),
            filters.SelectItems,
        ),
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS), 0, replace=False),
            filters.ExcludeItems,
        ),
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS, 2 * AMOUNT_OF_ITEMS), AMOUNT_SELECTED, replace=False),
            filters.SelectItems,
        ),
        (
            csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS))),
            np.random.choice(range(AMOUNT_OF_ITEMS, 2 * AMOUNT_OF_ITEMS), AMOUNT_SELECTED, replace=False),
            filters.ExcludeItems,
        ),
    ],
)
def test_exclude_items_shape_error(prediction_matrix, items, filter_class):
    myfilter = filter_class(items)
    with pytest.raises(ValueError):
        myfilter.apply_all(prediction_matrix)


@pytest.mark.parametrize(
    "filter_input, filter_class, name",
    [
        (np.random.randint(0, 2, size=(AMOUNT_OF_ITEMS,)).astype(bool), filters.ExcludeItems, "ExcludeItems"),
        (np.random.randint(0, 2, size=(AMOUNT_OF_ITEMS,)).astype(bool), filters.SelectItems, "SelectItems"),
    ],
)
def test_filter_items_str_repr(filter_input, filter_class, name):
    myfilter = filter_class(filter_input)

    assert name in myfilter.__str__()
    assert f"{filter_input}" in myfilter.__str__()
