import pytest

import recpack.preprocessing.filters as filters
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.data.matrix import InteractionMatrix


def test_dataframe_preprocessor_no_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    data_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.USER_IX].unique()
    )

    assert data_m.shape[0] == len(dataframe[InteractionMatrix.USER_IX].unique())
    assert data_m.shape[1] == len(dataframe[InteractionMatrix.ITEM_IX].unique())


def test_dataframe_preprocessor_no_filter_duplicates_dedupe(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    processor.add_filter(
        filters.Deduplicate(
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
        )
    )

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[2] = 12345

    dataframe.loc[dataframe.shape[0]] = row

    data_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.USER_IX].unique()
    )

    assert data_m.shape[0] == len(dataframe[InteractionMatrix.USER_IX].unique())
    assert data_m.shape[1] == len(dataframe[InteractionMatrix.ITEM_IX].unique())

    assert org_row[2] == data_m.timestamps[row[0], row[1]]
    assert not row[2] == data_m.timestamps[row[0], row[1]]


def test_dataframe_preprocessor_no_filter_duplicates_no_dedupe(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[2] = 12345

    dataframe.loc[dataframe.shape[0]] = row

    data_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.USER_IX].unique()
    )

    assert data_m.shape[0] == len(dataframe[InteractionMatrix.USER_IX].unique())
    assert data_m.shape[1] == len(dataframe[InteractionMatrix.ITEM_IX].unique())

    two_values = data_m.timestamps[row[0], row[1]]

    assert two_values.shape[0] == 2
    assert row[2] in two_values.values
    assert org_row[2] in two_values.values


def test_dataframe_preprocessor_id_mapping_w_multiple_dataframes(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )

    dataframe_2 = dataframe.copy()

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[0] = 666
    row[1] = 333

    dataframe.loc[dataframe.shape[0]] = row

    row_2 = org_row.copy()
    row_2[0] = 111
    row_2[1] = 555
    dataframe_2.loc[dataframe_2.shape[0]] = row_2
    # ---
    data_ms = processor.process_many(dataframe, dataframe_2)

    unique_users = set(dataframe[InteractionMatrix.USER_IX].unique()).union(
        dataframe_2[InteractionMatrix.USER_IX].unique()
    )
    unique_items = set(dataframe[InteractionMatrix.ITEM_IX].unique()).union(
        dataframe_2[InteractionMatrix.ITEM_IX].unique()
    )

    assert len(processor.item_id_mapping.keys()) == len(unique_items)
    assert len(processor.user_id_mapping.keys()) == len(unique_users)

    assert len(processor.item_id_mapping.keys()) != len(
        dataframe_2[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) != len(
        dataframe_2[InteractionMatrix.USER_IX].unique()
    )

    assert data_ms[0].shape[0] == len(unique_users)
    assert data_ms[0].shape[1] == len(unique_items)

    assert data_ms[0].shape[0] == data_ms[1].shape[0]
    assert data_ms[0].shape[1] == data_ms[1].shape[1]


def test_raises(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )

    with pytest.raises(RuntimeError):
        processor.map_users(dataframe)

    with pytest.raises(RuntimeError):
        processor.map_items(dataframe)


def test_dataframe_preprocessor_w_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    myfilter = filters.NMostPopular(
        3,
        InteractionMatrix.ITEM_IX,
    )

    processor.add_filter(myfilter)

    data_m = processor.process(dataframe)

    filtered_df = myfilter.apply(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        filtered_df[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        filtered_df[InteractionMatrix.USER_IX].unique()
    )

    assert data_m.shape[0] == len(filtered_df[InteractionMatrix.USER_IX].unique())
    assert data_m.shape[1] == len(filtered_df[InteractionMatrix.ITEM_IX].unique())


def test_add_filter(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    processor.add_filter(
        filters.MinUsersPerItem(
            3,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
        )
    )
    processor.add_filter(
        filters.NMostPopular(
            3,
            InteractionMatrix.ITEM_IX,
        )
    )

    processor.add_filter(
        filters.MinItemsPerUser(
            3,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
        ),
        index=1,
    )

    assert type(processor.filters[0]) == filters.MinUsersPerItem
    assert type(processor.filters[1]) == filters.MinItemsPerUser
    assert type(processor.filters[2]) == filters.NMostPopular
