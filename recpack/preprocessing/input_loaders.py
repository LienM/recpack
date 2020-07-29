from dataclasses import dataclass, field
from typing import List
import pandas

from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
from recpack.utils import logger

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
        logger.info(f"loading data from {self.file_name}")
        data = pandas.read_csv(self.file_name)

        min_users_filter = MinUsersPerItem(self.min_users_per_item, self.user_col, self.item_col, self.timestamp_col)
        min_items_filter = MinItemsPerUser(self.min_items_per_user, self.user_col, self.item_col, self.timestamp_col)

        filt_data = min_users_filter.apply(data)
        filt_data = min_items_filter.apply(filt_data)

        preprocessor = DataFramePreprocessor(self.item_col, self.user_col, self.timestamp_col, dedupe=self.dedupe)
        mat, = preprocessor.process(filt_data)

        logger.info(f"Data loaded, matrix shape = {mat.shape}")
        return mat

@dataclass
class CSVMetadataLoaderSimple:
    """Loads a metadata csv, and uses a list of fields as index for the matrix, no preprocessing happens on the string values
    This can be used for example to get a column per category / topic
    """
    file_name: str
    fields: List[str]
    item_col: str = field(default='item_id')
        
    def __post_init__(self):
        self.preprocessor = DataFramePreprocessor('metadata_token', self.item_col, dedupe=True)

    def load(self):
        data = pandas.read_csv(self.file_name)
        data.fillna("")
        # We use the values of the fields columns as tokens (without any processing/splitting/...)
        # and create a 'pseudo' interaction df, which we can process using the same methods we use for interactions.
        dataframes = []
        for field in self.fields:
            t = data[[self.item_col, field]].rename(columns={field: 'metadata_token'})
            dataframes.append(t)
        
        psuedo_interactions = dataframes[0]
        if len(dataframes) > 1:
            for additional_data in dataframes[1:]:
                psuedo_interactions = psuedo_interactions.append(additional_data)
        psuedo_interactions = psuedo_interactions[psuedo_interactions['metadata_token']!= ""]
        mat, = self.preprocessor.process(psuedo_interactions)

        return mat

