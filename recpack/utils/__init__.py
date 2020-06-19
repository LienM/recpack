import sys
import csv
import logging.config
from collections import defaultdict
import logging

import pandas as pd

import recpack.data_matrix

logger = logging.getLogger("recpack")
logger.setLevel(logging.INFO)


if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger():
    return logger


def groupby2(keys, values):
    """ A group by of separate lists where order doesn't matter. """
    multidict = defaultdict(list)
    for k, v in zip(keys, values):
        multidict[k].append(v)
    return multidict.items()


def to_tuple(el):
    """ Whether single element or tuple, always returns as tuple. """
    if type(el) == tuple:
        return el
    else:
        return (el, )


def dict_to_csv(d, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerow(d.values())


USER_KEY = "user"
ITEM_KEY = "item"
VALUE_KEY = "value"


def sparse_to_csv(m, path, values=True):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        coo = m.tocoo()
        if values:
            writer.writerow([USER_KEY, ITEM_KEY, VALUE_KEY])
            for u, i, v in zip(coo.row, coo.col, coo.data):
                writer.writerow([u, i, v])
        else:
            writer.writerow([USER_KEY, ITEM_KEY])
            for u, i in zip(coo.row, coo.col):
                writer.writerow([u, i])


def csv_to_sparse(path, values=True):
    df = pd.read_csv(path)
    if values:
        return recpack.data_matrix.DataM._create_values(df, ITEM_KEY, USER_KEY, VALUE_KEY)
    else:
        return recpack.data_matrix.DataM._create_values(df, ITEM_KEY, USER_KEY)
