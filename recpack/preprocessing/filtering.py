from abc import ABC, abstractmethod

import pandas as pd


class Filter(ABC):

    def __init__(self, user_id, item_id, timestamp_id=None):
        self.user_id = user_id
        self.item_id = item_id
        self.timestamp_id = timestamp_id

    @abstractmethod
    def apply(self, df: pd.DataFrame):
        pass


class MinUsersPerItem(Filter):

    def __init__(self, min_users_per_item, user_id, item_id, timestamp_id=None):
        self.min_ui = min_users_per_item

        super().__init__(user_id, item_id, timestamp_id)

    def apply(self, df):
        cnt_users_per_item = df[self.item_id].value_counts().reset_index()
        items_of_interest = cnt_users_per_item[cnt_users_per_item.index > self.min_ui]["index"].unique()

        return df[df[self.item_id].isin(items_of_interest)]


class NMostPopular(Filter):
    pass


class EventsSince(Filter):
    pass


class MinItemsPerUser(Filter):

    def __init__(self, min_items_per_user, user_id, item_id, timestamp_id=None):
        self.min_ui = min_items_per_user

        super().__init__(user_id, item_id, timestamp_id)

    def apply(self, df):
        cnt_items_per_user = df[self.user_id].value_counts().reset_index()
        users_of_interest = cnt_items_per_user[cnt_items_per_user.index > self.min_iu]["index"].unique()

        return df[df[self.user_id].isin(users_of_interest)]
