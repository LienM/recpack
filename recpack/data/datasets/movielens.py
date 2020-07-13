import pandas as pd

from recpack.data.data_source import DataSource
from recpack.preprocessing.filters import MinItemsPerUser
from recpack.data.data_matrix import DataM


class ML20MDataSource(DataSource):
    @property
    def data_name(self):
        return "ML"

    user_id = "userId"
    item_id = "movieId"
    value_id = "rating"

    def load_df(self, _path='/data/ml20m/ml-20m_ratings.csv'):
        df = pd.read_csv(_path)
        return df

    def get_preprocessor(self, min_iu=5):
        preprocessor = super().get_preprocessor()
        # preprocessor.add_filter(MinUsersPerItem(min_ui, self.user_id, self.item_id))
        preprocessor.add_filter(MinItemsPerUser(min_iu, self.user_id, self.item_id))
        return preprocessor

    def preprocess(self, min_rating=4):
        df = self.load_df()
        preferences = df[df[self.value_id] >= min_rating]
        preprocessor = self.get_preprocessor()
        viewData, = preprocessor.process(preferences)
        return DataM(viewData.binary_values)
