import numpy

import recpack.algorithms.retail_algorithms as retail_algorithms
import recpack.algorithms.true_baseline_algorithms as true_baseline_algorithms


def test_product_labeler_durable(pageviews, purchases, labels):
    """
    User 0 has purchased item 0,
    User 1 has purched item 4.
    Item 0 is durable, item 4 is not.
    """
    pl = retail_algorithms.ProductLabeler()
    pl.fit(labels, purchases)

    durables = pl.get_durable(pageviews)

    assert durables.shape == pageviews.shape
    assert len(durables.nonzero()[1]) == 1

    assert durables.nonzero()[1][0] == 0


def test_product_labeler_consumable(pageviews, purchases, labels):
    """
    User 0 has purchased item 0,
    User 1 has purched item 4.
    Item 0 is durable, item 4 is not.
    """

    pl = retail_algorithms.ProductLabeler()
    pl.fit(labels, purchases)

    consumables = pl.get_consumable(pageviews)

    assert consumables.shape == pageviews.shape
    assert len(consumables.nonzero()[1]) == 5

    assert consumables[0, 2] == 2


def test_filter_durable_goods(pageviews, purchases, labels):
    pl = retail_algorithms.ProductLabeler()
    pl.fit(labels, purchases)

    base_algo = true_baseline_algorithms.Popularity(1)

    true_algo = retail_algorithms.FilterDurableGoods(base_algo, pl)

    true_algo.fit(pageviews)

    res = true_algo.predict(purchases)

    # No recommendations because purchase was filtered
    assert len(res[0].nonzero()[1]) == 0

    # Recommend item 3 (most viewed)
    assert res[2, 3] == 1
    assert len(res[2].nonzero()[1]) == 1


def test_discount_durable_goods(pageviews, purchases, labels):
    pl = retail_algorithms.ProductLabeler()
    pl.fit(labels, purchases)

    base_algo = true_baseline_algorithms.Popularity(1)

    true_algo = retail_algorithms.DiscountDurableGoods(base_algo, pl, discount_value=0.5, K=1)

    true_algo.fit(pageviews)

    res = true_algo.predict(purchases)

    # Purchase was discounted  so -1
    assert len(res[0].nonzero()[1]) == 1
    numpy.testing.assert_almost_equal(res[0, 3], -0.5)

    # Recommend item 3 (most viewed)
    assert res[2, 3] != 0
    assert len(res[2].nonzero()[1]) == 1
