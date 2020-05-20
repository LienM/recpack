import sys
import logging.config
from collections import defaultdict
import logging

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
