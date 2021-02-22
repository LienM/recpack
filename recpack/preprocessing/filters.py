from typing import List
from abc import ABC, abstractmethod

import pandas as pd

# TODO Improve interface so that arguments known to the Preprocessor don't
#   have to be duplidated.
# TODO: Make the names uniform.
#   In other places we have started using _ix io _id as suffix


class Filter(ABC):
    def apply_all(self, *dfs: pd.DataFrame) -> List[pd.DataFrame]:
        ret = list()
        for df in dfs:
            ret.append(self.apply(df))
        return ret

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Filter to the DataFrame passed.

        :param df: DataFrame to filter
        :type df: pd.DataFrame
        """
        raise NotImplementedError

    def __str__(self):
        attrs = self.__dict__
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"


class MinUsersPerItem(Filter):
    def __init__(
        self,
        min_users_per_item: int,
        item_ix: str,
        user_ix: str,
        count_duplicates: bool = False,
    ):
        """
        Require that a minimum number of users has interacted with an item.

        :param min_users_per_item: Minimum number of users required.
        :type min_users_per_item: int
        :param item_ix: Name of the column in which item identifiers are listed.
        :type item_ix: str
        :param user_ix: Name of the column in which user identifiers are listed.
        :type user_ix: str
        :param count_duplicates: Count multiple interactions with the same user, defaults to False
        :type count_duplicates: bool
        """
        self.item_ix = item_ix
        self.user_ix = user_ix

        self.min_ui = min_users_per_item
        self.count_duplicates = count_duplicates

    def apply(self, df):
        iids = (
            df[self.item_ix]
            if self.count_duplicates
            else df.drop_duplicates([self.user_ix, self.item_ix])[self.item_ix]
        )
        cnt_users_per_item = iids.value_counts()
        items_of_interest = list(
            cnt_users_per_item[cnt_users_per_item >= self.min_ui].index
        )

        return df[df[self.item_ix].isin(items_of_interest)]


class NMostPopular(Filter):
    def __init__(self, N: int, item_ix: str):
        """
        Retain only the N most popular items.
        Note: All interactions count, also if a user interacted with the same item
        1000 times.

        :param N: Number of items to retain.
        :type N: int
        :param item_ix: Name of the column in which item identifiers are listed.
        :type item_ix: str
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        """
        self.N = N
        self.item_ix = item_ix

    def apply(self, df):
        cnt_users_per_item = df[self.item_ix].value_counts(sort=True, ascending=False)

        items_of_interest = list(cnt_users_per_item[0 : self.N].index)

        return df[df[self.item_ix].isin(items_of_interest)]


class NMostRecent(Filter):
    """Select only events on the N most recently visited items.

    If there is a tie, the first occuring item in the dataframe is retained.

    :param N: Number of items to retain.
    :type N: int
    :param item_ix: Name of the column in which item identifiers are listed.
    :type item_ix: str
    :param timestamp_ix: Name of the column in which timestamps are listed, defaults to None
    :type timestamp_ix: str, optional
    """

    def __init__(self, N: int, item_ix: str, timestamp_ix: str):
        self.N = N
        self.item_ix = item_ix
        self.timestamp_ix = timestamp_ix

    def apply(self, df):
        sorted_df = df[[self.item_ix, self.timestamp_ix]].sort_values(
            by=self.timestamp_ix, ascending=False
        )
        ids = sorted_df[[self.item_ix]].drop_duplicates()

        return df[df[self.item_ix].isin(ids[: self.N][self.item_ix])]


class MinItemsPerUser(Filter):
    def __init__(
        self,
        min_items_per_user: int,
        item_ix: str,
        user_ix: str,
        count_duplicates: bool = False,
    ):
        """
        Require that a user has interacted with a minimum number of items.

        :param min_items_per_user: Minimum number of items required.
        :type min_items_per_user: int
        :param item_ix: Name of the column in which item identifiers are listed.
        :type item_ix: str
        :param user_ix: Name of the column in which user identifiers are listed.
        :type user_ix: str
        :param count_duplicates: Count multiple interactions with the same item, defaults to False
        :type count_duplicates: bool
        """
        self.min_iu = min_items_per_user
        self.count_duplicates = count_duplicates

        self.item_ix = item_ix
        self.user_ix = user_ix

    def apply(self, df):
        uids = (
            df[self.user_ix]
            if self.count_duplicates
            else df.drop_duplicates([self.user_ix, self.item_ix])[self.user_ix]
        )
        cnt_items_per_user = uids.value_counts()
        users_of_interest = list(
            cnt_items_per_user[cnt_items_per_user >= self.min_iu].index
        )

        return df[df[self.user_ix].isin(users_of_interest)]


class MinRating(Filter):
    """
    Create a dataframe of only ratings above min_rating.
    This filter is used to turn a rating dataset into an interaction dataset.

    After filtering, the rating_ix column will also be removed from the dataframe.

    :param min_rating: The minimum rating for a rating to be considered an interaction.
    :type min_rating: int
    :param rating_ix: the column that will contain ratings in the dataframe
    :type str:
    """

    def __init__(
        self,
        min_rating: int,
        rating_ix: str,
    ):
        self.rating_ix = rating_ix
        self.min_rating = min_rating

    def apply(self, df):
        return df[df[self.rating_ix] >= self.min_rating].drop(columns=self.rating_ix)


class Deduplicate(Filter):
    """Deduplicate entries with the same user and item.

    Removes all but one of each user-item pair in the dataframe.
    If timestamps are available, the first interaction is kept.
    :param item_ix: Name of the column in which item identifiers are listed.
    :type item_ix: str
    :param user_ix: Name of the column in which user identifiers are listed.
    :type user_ix: str
    :param timestamp_ix: Name of the column in which timestamps are listed, defaults to None
    :type timestamp_ix: str, optional
    """

    def __init__(
        self,
        item_ix: str,
        user_ix: str,
        timestamp_ix: str = None,
    ):
        self.item_ix = item_ix
        self.user_ix = user_ix
        self.timestamp_ix = timestamp_ix

    def apply(self, df):
        if self.timestamp_ix is not None:
            df = df.sort_values(self.timestamp_ix)
        return df.drop_duplicates([self.user_ix, self.item_ix], keep="first")
