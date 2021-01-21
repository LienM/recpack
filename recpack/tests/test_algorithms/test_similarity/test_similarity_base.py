from recpack.algorithms import ItemKNN
import warnings


def test_check_fit_complete(pageviews):
    # Set a row to 0, so it won't have any neighbours
    pv_copy = pageviews.copy()
    pv_copy[:, 4] = 0

    a = ItemKNN(2)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        a.fit(pv_copy)
        # The algorithm might also throw a warning
        assert len(w) >= 1

        assert "1 items" in str(w[-1].message)
