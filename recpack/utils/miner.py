""" Utilities for pattern mining (used by HO-EASE) """

from collections import defaultdict
import itertools

from tqdm.auto import tqdm

from recpack.utils.monitor import Monitor


def eclat(tidmap, prefix=[], minsup=1, size=3):
    if size == 0:
        return []

    output = list()
    tidlist = list(sorted(tidmap.items(), key=lambda x: len(x[1]), reverse=True))

    progress = None
    if len(prefix) == 0:
        progress = tqdm(desc='{' + ', '.join(map(str, prefix)) + '}', total=len(tidlist), leave=True)

    while tidlist:
        if progress:
            progress.update()
        item, tids = tidlist.pop()
        if len(tids) < minsup:
            continue

        if size == 1:
            output.append(frozenset(prefix + [item]))

        suffix = dict()
        for other, othertids in tidlist:
            newtids = tids & othertids
            if len(newtids) < minsup:
                continue

            suffix[other] = newtids

        new_itemsets = eclat(suffix, prefix + [item], minsup=minsup, size=size-1)
        output.extend(new_itemsets)

    return output


def process_itemsets(tidmap, itemsets):
    output = defaultdict(int)

    for itemset in tqdm(itemsets):
        for subset in itertools.combinations(itemset, 2):
            subset = frozenset(subset)
            (other,) = itemset - subset

            othertids = tidmap[other]
            tidsA, tidsB = (tidmap[item] for item in subset)
            tidsAB = tidsA & tidsB

            conf = len(tidsAB & othertids) / len(tidsAB)
            infl1 = abs(conf - len(tidsA & othertids) / len(tidsA))
            infl2 = abs(conf - len(tidsB & othertids) / len(tidsB))
            infl = min(infl1, infl2)
            output[subset] += infl

    return list(output.items())


def calculate_itemsets(X, minsup=2, amount=None):
    monitor = Monitor("Itemsets")

    monitor.update("tidmap")
    tidmap = defaultdict(set)
    rows, cols = X.nonzero()
    for row, col in zip(rows, cols):
        tidmap[col].add(row)

    monitor.update("eclat")
    itemsets = eclat(tidmap, minsup=minsup, size=3)

    print("Amount of itemsets:", len(itemsets))

    monitor.update("Calculate influence")
    itemsets = process_itemsets(tidmap, itemsets)

    monitor.update("Sort and trim")
    ordered = sorted(itemsets, key=lambda x: x[1], reverse=True)
    trimmed = [i for i, s in ordered]

    if amount:
        return trimmed[:amount]

    return trimmed
