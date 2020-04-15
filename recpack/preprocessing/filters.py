from abc import ABC, abstractmethod

import pandas as pd

# TODO Integrate into CLI


class Filter(ABC):

    def __init__(self, user_id, item_id, timestamp_id=None):
        self.user_id = user_id
        self.item_id = item_id
        self.timestamp_id = timestamp_id

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Filter to the DataFrame passed.

        :param df: DataFrame to filter
        :type df: pd.DataFrame
        """
        pass


class MinUsersPerItem(Filter):

    def __init__(self, min_users_per_item: int, user_id: str, item_id: str, timestamp_id=None):
        """
        Require that a minimum number of users has interacted with an item.

        :param min_users_per_item: Minimum number of users required.
        :type min_users_per_item: int
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param item_id: Name of the column in which item identifiers are listed.
        :type item_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        """
        self.min_ui = min_users_per_item

        super().__init__(user_id, item_id, timestamp_id)

    def apply(self, df):
        cnt_users_per_item = df[self.item_id].value_counts()
        items_of_interest = list(cnt_users_per_item[cnt_users_per_item >= self.min_ui].index)

        return df[df[self.item_id].isin(items_of_interest)]


class NMostPopular(Filter):
    def __init__(self, N: int, user_id: str, item_id: str, timestamp_id=None):
        """
        Retain only the N most popular items.

        :param N: Number of items to retain.
        :type N: int
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param item_id: Name of the column in which item identifiers are listed.
        :type item_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        """
        self.N = N

        super().__init__(user_id, item_id, timestamp_id)

    def apply(self, df):
        cnt_users_per_item = df[self.item_id].value_counts(sort=True, ascending=False)

        items_of_interest = list(cnt_users_per_item[0:self.N].index)

        return df[df[self.item_id].isin(items_of_interest)]


class EventsSince(Filter):
    pass


class MinItemsPerUser(Filter):

    def __init__(self, min_items_per_user: int, user_id: str, item_id: str, timestamp_id=None):
        """
        Require that a user has interacted with a minimum number of item.

        :param min_items_per_user: Minimum number of items required.
        :type min_items_per_user: int
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param item_id: Name of the column in which item identifiers are listed.
        :type item_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        """
        self.min_iu = min_items_per_user

        super().__init__(user_id, item_id, timestamp_id)

    def apply(self, df):
        cnt_items_per_user = df[self.user_id].value_counts()
        users_of_interest = list(cnt_items_per_user[cnt_items_per_user >= self.min_iu].index)

        return df[df[self.user_id].isin(users_of_interest)]
