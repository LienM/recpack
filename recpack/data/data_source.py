from recpack.preprocessing.preprocessors import DataFramePreprocessor

from recpack.utils import to_tuple


class DataSource(object):
    user_id = "user"
    item_id = "item"
    value_id = None
    timestamp_id = None

    def __init__(self):
        super().__init__()

    @property
    def data_name(self):
        raise NotImplementedError("Need to override data_name")

    def get_params(self):
        params = super().get_params() if hasattr(super(), "get_params") else dict()
        params["data_source"] = self.data_name
        return params

    def load_df(self):
        raise NotImplementedError("Need to override `load_df` or `preprocess`")

    @property
    def preprocessor(self):
        preprocessor = DataFramePreprocessor(
            self.item_id, self.user_id, self.value_id, self.timestamp_id, dedupe=True
        )
        return preprocessor

    def preprocess(self):
        """
        Return one or two datasets of type DataM
        """
        dfs = to_tuple(self.load_df())
        datasets = self.preprocessor.process(*dfs)
        return datasets

    @property
    def item_id_mapping(self):
        return self.preprocessor.item_id_mapping

    @property
    def user_id_mapping(self):
        return self.preprocessor.user_id_mapping
