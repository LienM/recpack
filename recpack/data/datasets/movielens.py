import pandas as pd

from recpack.data.data_source import DataSource
from recpack.preprocessing.filters import MinItemsPerUser
from recpack.data.data_matrix import DataM


class ML20MDataSource(DataSource):
    @property
    def name(self):
        return "ML"

    user_id = "userId"
    item_id = "movieId"
    value_id = "rating"

    def load_df(self, path=None):
        df = pd.read_csv(path)
        return df

    def preprocess(self, path, min_rating=4, min_iu=5):
        df = self.load_df(path=path)
        preferences = df[df[self.value_id] >= min_rating]
        preprocessor = self.preprocessor
        preprocessor.add_filter(MinItemsPerUser(min_iu, self.user_id, self.item_id))

        view_data, = preprocessor.process(preferences)
        return DataM(view_data.binary_values)
