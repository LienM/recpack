import numpy as np
import pandas as pd


from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
from recpack.data.matrix import InteractionMatrix, to_binary
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.util import to_tuple

# TODO Refactor so that it downloads the datasets and names the fields consistently


def load_citeulike(self, path, min_ui=5, min_iu=1):

    # Different on purpose from the InteractionMatrix ones to avoid name colisions.
    USER_IX = "user"
    ITEM_IX = "item"

    # TODO: Download if not at path

    # Load from path
    u_i_pairs = []
    with open(path, "r") as f:

        for user, line in enumerate(f.readlines()):
            item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
            items = line.strip("\n").split(" ")[1:]
            assert len(items) == int(item_cnt)

            for item in items:
                assert item.isdecimal()  # Make sure the identifiers are correct.
                u_i_pairs.append((user, item))

    df = pd.DataFrame(u_i_pairs, columns=[USER_IX, ITEM_IX])

    # Process to InteractionMatrix object
    preprocessor = DataFramePreprocessor(ITEM_IX, USER_IX, dedupe=True)
    preprocessor.add_filter(MinItemsPerUser(min_iu, USER_IX, ITEM_IX))
    preprocessor.add_filter(MinUsersPerItem(min_ui, USER_IX, ITEM_IX))

    return preprocessor.process(df)


def load_ml_20m(path, min_rating=4, min_ui=5, min_iu=1):
    USER_IX = "userId"
    ITEM_IX = "itemId"
    TIMESTAMP_IX = "timestamp"
    RATING_IX = "rating"
    # Also has a rating, which we will ignore for now
    # If we add a RatingMatrix it should be added as an option to load

    # TODO: download from path

    df = pd.read_csv(path)

    # keep only the rows that have a high enough rating
    interactions = df[df[RATING_IX] >= min_rating]

    # Process to InteractionMatrix object
    preprocessor = DataFramePreprocessor(
        ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX, dedupe=True
    )
    preprocessor.add_filter(MinItemsPerUser(min_iu, USER_IX, ITEM_IX))
    preprocessor.add_filter(MinUsersPerItem(min_ui, USER_IX, ITEM_IX))

    data = preprocessor.process(interactions)
    return to_binary(data)


def load_rsc_2015(path, nrows=None, min_ui=5, min_iu=1):
    # On purpose different from the InteractionMatrix ones, to avoid column colision.
    USER_IX = "session"
    ITEM_IX = "item"
    TIMESTAMP_IX = "timestamp"

    df = pd.read_csv(
        path,
        names=[USER_IX, TIMESTAMP_IX, ITEM_IX],
        dtype={USER_IX: np.int64, TIMESTAMP_IX: np.str, ITEM_IX: np.int64},
        parse_dates=[TIMESTAMP_IX],
        usecols=[0, 1, 2],
        nrows=nrows,
    )

    df[TIMESTAMP_IX] = (
        df[TIMESTAMP_IX].astype(int) / 1e9
    )  # pandas datetime -> seconds from epoch

    # Process to dataframe
    preprocessor = DataFramePreprocessor(
        ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX, dedupe=True
    )

    filter_users = MinItemsPerUser(
        min_iu,
        item_id=ITEM_IX,
        user_id=USER_IX,
        timestamp_id=TIMESTAMP_IX,
        count_duplicates=True,
    )
    filter_items = MinUsersPerItem(
        min_ui,
        item_id=ITEM_IX,
        user_id=USER_IX,
        timestamp_id=TIMESTAMP_IX,
        count_duplicates=True,
    )

    # add the filters
    preprocessor.add_filter(filter_users)
    preprocessor.add_filter(filter_items)
    # Apply user filter a second time to remove users
    # with too few items in their history after the items filter.
    preprocessor.add_filter(filter_users)

    return preprocessor.process(df)
