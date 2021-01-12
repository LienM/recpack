from recpack.preprocessing.preprocessors import DataFramePreprocessor

from recpack.util import to_tuple


class DataSource(object):
    """Load and preprocess a dataset into a `DataM` object

    TODO: the interface as is, is not very logical in use.
        eg. not possible to pass a path to the file.

    :param object: [description]
    :type object: [type]
    :raises NotImplementedError: [description]
    :raises NotImplementedError: [description]
    :return: [description]
    :rtype: [type]
    """

    user_id = "user"
    item_id = "item"
    value_id = None
    timestamp_id = None

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        raise NotImplementedError("Need to override name")

    def get_params(self):
        params = super().get_params() if hasattr(super(), "get_params") else dict()
        params["data_source"] = self.name
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
        Return a dataset of type DataM
        """
        df = to_tuple(self.load_df())
        data_m = self.preprocessor.process(df)
        return data_m

    @property
    def item_id_mapping(self):
        return self.preprocessor.item_id_mapping

    @property
    def user_id_mapping(self):
        return self.preprocessor.user_id_mapping
