from recpack.preprocessing.preprocessors import DataFramePreprocessor


def test_dataframe_preprocessor_no_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor("iid", "uid", "timestamp", dedupe=False)

    data_m = processor.process(dataframe)

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

    data_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe["iid"].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe["uid"].unique()
    )

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

    data_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe["iid"].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe["uid"].unique()
    )

    assert data_m.shape[0] == len(dataframe["uid"].unique())
    assert data_m.shape[1] == len(dataframe["iid"].unique())

    two_values = data_m.timestamps[row[0], row[1]]

    assert two_values.shape[0] == 2
    assert row[2] in two_values.values
    assert org_row[2] in two_values.values


def test_dataframe_preprocessor_filter_no_duplicates(dataframe):
    pass


def test_dataframe_preprocessor_id_mapping_w_multiple_dataframes(dataframe):
    pass
