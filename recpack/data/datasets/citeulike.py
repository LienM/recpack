import pandas as pd

from recpack.data.data_source import DataSource
from recpack.preprocessing.filters import MinItemsPerUser
from recpack.data.data_matrix import DataM


class CiteULike(DataSource):
    @property
    def name(self):
        return "citeulike"

    user_id = "user_id"
    item_id = "item_id"

    @classmethod
    def load_df(cls, data_file):
        # TODO Download data
        u_i_pairs = []
        with open(data_file, "r") as f:
            for user, line in enumerate(f.readlines()):
                items = line.strip("\n").split(" ")[1:]  # First element is a count
                item_cnt = line.strip("\n").split(" ")[0]
                assert len(items) == int(item_cnt)
                for item in items:
                    assert item.isdecimal()  # Make sure the identifiers are correct.
                    u_i_pairs.append((user, item))

        return pd.DataFrame(u_i_pairs, columns=[cls.user_id, cls.item_id])

    def preprocess(self, data_file, min_iu=5):
        df = self.load_df(data_file)
        preprocessor = self.preprocessor
        preprocessor.add_filter(MinItemsPerUser(min_iu, self.user_id, self.item_id))

        view_data = preprocessor.process(df)
        return view_data
