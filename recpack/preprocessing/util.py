# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert



def rescale_id_space(ids, id_mapping=None):
    """
    Map the given ids to indices,
    if id_mapping is not None, use that as start, and add new values
    """
    counter = 0

    if id_mapping is not None and len(id_mapping) > 0:
        counter = max(id_mapping.values()) + 1
    else:
        id_mapping = {}
    for val in ids:
        if val not in id_mapping:
            id_mapping[val] = counter
            counter += 1

    return id_mapping
