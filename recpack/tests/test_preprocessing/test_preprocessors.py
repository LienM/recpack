from recpack.preprocessing.preprocessors import DataFramePreprocessor


def test_dataframe_preprocessor_no_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor("iid", "uid", "timestamp", dedupe=False)

    sp_mat = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(dataframe["iid"].unique())
    assert len(processor.user_id_mapping.keys()) == len(dataframe["uid"].unique())

    assert sp_mat.shape[0] == len(dataframe["uid"].unique())
    assert sp_mat.shape[1] == len(dataframe["iid"].unique())


def test_dataframe_preprocessor_no_filter_duplicates_dedupe(dataframe):
    pass

def test_dataframe_preprocessor_no_filter_duplicates_no_dedupe(dataframe):
    pass

def test_dataframe_preprocessor_filter_no_duplicates(dataframe):
    pass

def test_dataframe_preprocessor_id_mapping_w_multiple_dataframes(dataframe):
    pass
