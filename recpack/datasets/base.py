# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Module responsible for handling datasets."""

import os
import pandas as pd
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
from recpack.preprocessing.filters import (
    Filter,
)
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.preprocessors import DataFramePreprocessor


def _fetch_remote(url: str, filename: str) -> str:
    """Fetch data from remote url and save locally

    :param url: url to fetch data from
    :type url: str
    :param filename: Path to save file to
    :type filename: str
    :return: The filename where data was saved
    :rtype: str
    """
    urlretrieve(url, filename)
    return filename


class Dataset:
    """Represents a collaborative filtering dataset,
    containing users who interacted in some way with a set of items.

    Every Dataset has a set of preprocessing defaults,
    i.e. filters that are commonly applied to the dataset before use in recommendation algorithms.
    These can be disabled and a different set of filters can be applied.

    A Dataset is transformed into an InteractionMatrix

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional
    """

    USER_IX = None
    """Name of the column in the DataFrame with user identifiers"""
    ITEM_IX = None
    """Name of the column in the DataFrame with item identifiers"""
    TIMESTAMP_IX = None
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = None
    """Default filename that will be used if it is not specified by the user."""

    def __init__(self, path: str = "data", filename: str = None, use_default_filters=True):
        self.filename = filename
        if not self.filename:
            if self.DEFAULT_FILENAME:
                self.filename = self.DEFAULT_FILENAME
            else:
                raise ValueError("No filename specified, and no default known.")

        self.path = path
        self.preprocessor = DataFramePreprocessor(self.ITEM_IX, self.USER_IX, self.TIMESTAMP_IX)
        if use_default_filters:
            for f in self._default_filters:
                self.add_filter(f)

        self._ensure_path_exists()

    @property
    def file_path(self):
        """The fully classified path to the file from which dataset will be loaded."""
        # TODO: correctness check?
        return os.path.join(self.path, self.filename)

    def _ensure_path_exists(self):
        """Constructs directory if path is not present on disk."""
        p = Path(self.path)
        p.mkdir(exist_ok=True)

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters to apply to a dataframe.

        Should be defined for a dataset if there is default preprocessing.

        :return: A list of filters
        :rtype: List[Filter]
        """
        return []

    def add_filter(self, _filter: Filter, index=None):
        """Add a filter to be applied when loading the data.

        If the index is specified, the filter is inserted at the specified index.
        Otherwise it is appended.

        :param _filter: Filter to be applied to the loaded DataFrame
                    processing to interaction matrix.
        :type _filter: Filter
        :param index: The index to insert the filter at,
            None will append the filter. Defaults to None
        :type index: int

        """
        self.preprocessor.add_filter(_filter, index=index)

    def fetch_dataset(self, force=False):
        """Check if dataset is present, if not download

        :param force: If True, dataset will be downloaded,
                even if the file already exists.
                Defaults to False.
        :type force: bool, optional
        """
        if not os.path.exists(self.file_path) or force:
            self._download_dataset()

    def _download_dataset(self):
        raise NotImplementedError("Should still be implemented")

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError("Needs to be implemented")

    def load(self) -> InteractionMatrix:
        """Loads data into an InteractionMatrix object.

        Data is loaded into a DataFrame using the `_load_dataframe` function.
        Resulting DataFrame is parsed into an `InteractionMatrix` object.
        During parsing the filters are applied in order.

        :return: The resulting InteractionMatrix
        :rtype: InteractionMatrix
        """
        df = self._load_dataframe()

        return self.preprocessor.process(df)
