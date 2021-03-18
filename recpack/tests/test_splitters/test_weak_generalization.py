import numpy as np
import pytest

import recpack.splitters.scenarios as scenarios


def test_weak_generalization_invalid_perc(data_m):
    with pytest.raises(AssertionError):
        scenarios.WeakGeneralization(
            0.7, frac_interactions_validation=0.5, validation=True
        )


@pytest.mark.parametrize("frac_interactions_train", [0.5, 0.7])
def test_weak_generalization_split(data_m, frac_interactions_train):

    scenario = scenarios.WeakGeneralization(frac_interactions_train)
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data

    # Test no validation data
    assert not scenario.validation_data
    # Test approximately correct split
    perc_interactions_test = 1 - frac_interactions_train

    np.testing.assert_almost_equal(
        tr.values.nnz / data_m.values.nnz, frac_interactions_train, decimal=2
    )
    np.testing.assert_almost_equal(
        te_data_out.values.nnz / data_m.values.nnz, perc_interactions_test, decimal=2
    )

    # te_data_in =~ tr (except users that had no interactions in te_data_out)
    assert set(tr.indices[0]).intersection(te_data_in.indices[0])
    # Users have interactions in both
    assert te_data_out.active_users == te_data_in.active_users


@pytest.mark.parametrize(
    "frac_interactions_train, frac_interactions_validation", [(0.5, 0.25), (0.25, 0.25)]
)
def test_weak_generalization_split_w_validation(
    data_m, frac_interactions_train, frac_interactions_validation
):

    scenario = scenarios.WeakGeneralization(
        frac_interactions_train,
        frac_interactions_validation=frac_interactions_validation,
        validation=True,
    )
    scenario.split(data_m)

    tr = scenario.training_data
    te_data_in, te_data_out = scenario.test_data
    val_data_in, val_data_out = scenario.validation_data
    # Test approximately correct split
    perc_interactions_test = 1 - frac_interactions_train - frac_interactions_validation

    np.testing.assert_almost_equal(
        tr.values.nnz / data_m.values.nnz, frac_interactions_train, decimal=2
    )
    np.testing.assert_almost_equal(
        te_data_out.values.nnz / data_m.values.nnz, perc_interactions_test, decimal=2
    )
    np.testing.assert_almost_equal(
        val_data_out.values.nnz / data_m.values.nnz,
        frac_interactions_validation,
        decimal=2,
    )

    # te_data_in =~ tr (except users that had no interactions in te_data_out)
    assert set(tr.indices[0]).intersection(te_data_in.indices[0])
    # Users have interactions in both
    assert te_data_out.active_users == te_data_in.active_users
    assert val_data_in.active_users == val_data_out.active_users
    # tr = val_data_in
    assert val_data_in.values.nnz == tr.values.nnz
