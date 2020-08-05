import csv
import logging

import pandas as pd

from recpack.experiments.globals import USER_KEY, ITEM_KEY, VALUE_KEY
from recpack.util import df_to_sparse

logger = logging.getLogger("recpack")


def dict_to_csv(d, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerow(d.values())


class ItemCSVWriter(object):
    def __init__(self, item_id_mapping=None):
        super().__init__()
        self.item_id_mapping = item_id_mapping
        self.reverse_item_id_mapping = dict()
        if self.item_id_mapping is not None:
            pass

    def reverse_map_item_id(self, iid):
        return self.reverse_item_id_mapping.get(iid, iid)


class UserCSVWriter(object):
    def __init__(self, user_id_mapping=None):
        super().__init__()
        self.user_id_mapping = user_id_mapping
        self.reverse_user_id_mapping = dict()
        if self.user_id_mapping is not None:
            pass

    def reverse_map_user_id(self, uid):
        return self.reverse_user_id_mapping.get(uid, uid)


class InteractionsCSVWriter(UserCSVWriter, ItemCSVWriter):
    def __init__(self, user_id_mapping=None, item_id_mapping=None):
        UserCSVWriter.__init__(self, user_id_mapping=user_id_mapping)
        ItemCSVWriter.__init__(self, item_id_mapping=item_id_mapping)

    def reverse_map_user_item_id(self, uid, iid):
        return self.reverse_map_user_id(uid), self.reverse_map_item_id(iid)

    def sparse_to_csv(self, m, path, values=True):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            coo = m.tocoo()
            if values:
                writer.writerow([USER_KEY, ITEM_KEY, VALUE_KEY])
                for u, i, v in zip(coo.row, coo.col, coo.data):
                    uid, iid = self.reverse_map_user_item_id(u, i)
                    writer.writerow([uid, iid, v])
            else:
                writer.writerow([USER_KEY, ITEM_KEY])
                for u, i in zip(coo.row, coo.col):
                    uid, iid = self.reverse_map_user_item_id(u, i)
                    writer.writerow([uid, iid])


def csv_to_sparse(path, values=True):
    df = pd.read_csv(path)
    if values:
        return df_to_sparse(df, ITEM_KEY, USER_KEY, VALUE_KEY)
    else:
        return df_to_sparse(df, ITEM_KEY, USER_KEY)
