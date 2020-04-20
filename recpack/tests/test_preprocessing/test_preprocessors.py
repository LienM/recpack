from recpack.preprocessing.preprocessors import DataFramePreprocessor


def test_dataframe_preprocessor_no_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor("iid", "uid", "timestamp", dedupe=False)

    data_ms = processor.process(dataframe)

    data_m = data_ms[0]

    assert len(processor.item_id_mapping.keys()) == len(dataframe["iid"].unique())
    assert len(processor.user_id_mapping.keys()) == len(dataframe["uid"].unique())

    assert data_m.shape[0] == len(dataframe["uid"].unique())
    assert data_m.shape[1] == len(dataframe["iid"].unique())


def test_dataframe_preprocessor_no_filter_duplicates_dedupe(dataframe):
    processor = DataFramePreprocessor("iid", "uid", "timestamp", dedupe=True)

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[2] = 12345

    dataframe.loc[dataframe.shape[0]] = row

    data_ms = processor.process(dataframe)

    data_m = data_ms[0]

    assert len(processor.item_id_mapping.keys()) == len(dataframe["iid"].unique())
    assert len(processor.user_id_mapping.keys()) == len(dataframe["uid"].unique())

    assert data_m.shape[0] == len(dataframe["uid"].unique())
    assert data_m.shape[1] == len(dataframe["iid"].unique())

    assert org_row[2] == data_m.timestamps[row[0], row[1]]
    assert not row[2] == data_m.timestamps[row[0], row[1]]


def test_dataframe_preprocessor_no_filter_duplicates_no_dedupe(dataframe):
    processor = DataFramePreprocessor("iid", "uid", "timestamp", dedupe=False)

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[2] = 12345

    dataframe.loc[dataframe.shape[0]] = row

    data_ms = processor.process(dataframe)

    data_m = data_ms[0]

    assert len(processor.item_id_mapping.keys()) == len(dataframe["iid"].unique())
    assert len(processor.user_id_mapping.keys()) == len(dataframe["uid"].unique())

    assert data_m.shape[0] == len(dataframe["uid"].unique())
    assert data_m.shape[1] == len(dataframe["iid"].unique())

    two_values = data_m.timestamps[row[0], row[1]]

    assert two_values.shape[0] == 2
    assert row[2] in two_values.values
    assert org_row[2] in two_values.values


# def test_dataframe_preprocessor_filter_no_duplicates(dataframe):
#     pass


def test_dataframe_preprocessor_id_mapping_w_multiple_dataframes(dataframe):
    processor = DataFramePreprocessor("iid", "uid", "timestamp", dedupe=False)

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
    data_ms = processor.process(dataframe, dataframe_2)

    unique_users = set(dataframe["uid"].unique()).union(dataframe_2["uid"].unique())
    unique_items = set(dataframe["iid"].unique()).union(dataframe_2["iid"].unique())

    assert len(processor.item_id_mapping.keys()) == len(unique_items)
    assert len(processor.user_id_mapping.keys()) == len(unique_users)

    assert len(processor.item_id_mapping.keys()) != len(dataframe_2["iid"].unique())
    assert len(processor.user_id_mapping.keys()) != len(dataframe_2["uid"].unique())

    assert data_ms[0].shape[0] == len(unique_users)
    assert data_ms[0].shape[1] == len(unique_items)

    assert data_ms[0].shape[0] == data_ms[1].shape[0]
    assert data_ms[0].shape[1] == data_ms[1].shape[1]
