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
    def load_data(cls, data_folder_path):
        # TODO download dataset?
        u_i_pairs = []
        with open(f"{data_folder_path}/users.dat", "r") as f:
            for user, line in enumerate(f.readlines()):
                items = line.strip("\n").split(" ")[1:]  # First element is a count
                item_cnt = line.strip("\n").split(" ")[0]
                assert len(items) == int(item_cnt)
                for item in items:
                    assert item.isdecimal()  # Make sure the identifiers are correct.
                    u_i_pairs.append((user, item))

        # item_tag_pairs = []
        # with open(f"{data_folder_path}/tag-item.dat", "r") as f:
        #     for item, line in enumerate(f.readlines()):
        #         tags = line.strip("\n").split("")[1:]  # First element is a count
        #         tag_cnt = line.strip("\n").split("")[0]
        #         assert len(tags) == tag_cnt
        #         for tag in tags:
        #             assert tag.isdecimal()  # Make sure the identifiers are correct.
        #             item_tag_pairs.append((item, tag))

        return pd.DataFrame(u_i_pairs, columns=[cls.user_id, cls.item_id])

    # def preprocess(self, min_rating=4, min_iu=5):
    #     df = self.load_df()
    #     preferences = df[df[self.value_id] >= min_rating]
    #     preprocessor = self.preprocessor
    #     preprocessor.add_filter(MinItemsPerUser(min_iu, self.user_id, self.item_id))

    #     (view_data,) = preprocessor.process(preferences)
    #     return DataM(view_data.binary_values)
