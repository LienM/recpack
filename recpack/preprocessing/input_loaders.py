from dataclasses import dataclass, field
import logging
import pandas

from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
class InputLoader:
    def load(self):
        pass

@dataclass
class CSVInteractionsLoader:
    file_name: str
    item_col: str = field(default='item_id')
    user_col: str = field(default='user_id')
    timestamp_col: str = field(default='timestamp')
    min_users_per_item: int = field(default=1)
    min_items_per_user: int = field(default=1)
    dedupe: bool = field(default=True)

    def load(self):
        logging.info(f"loading data from {self.file_name}")
        data = pandas.read_csv(self.file_name)
        # data = data.sample(100_000)
        min_users_filter = MinUsersPerItem(self.min_users_per_item, self.user_col, self.item_col, self.timestamp_col)
        min_items_filter = MinItemsPerUser(self.min_items_per_user, self.user_col, self.item_col, self.timestamp_col)

        filt_data = min_users_filter.apply(data)
        filt_data = min_items_filter.apply(filt_data)

        preprocessor = DataFramePreprocessor(self.item_col, self.user_col, self.timestamp_col, dedupe=self.dedupe)
        mat, = preprocessor.process(filt_data)

        logging.info(f"Data loaded, matrix shape = {mat.shape}")
        return mat
