import numpy as np
import pytest
import torch

from recpack.algorithms import NeuMFMLPOnly
from recpack.algorithms.neumf import NeuMFMLPModule

from recpack.tests.test_algorithms.util import assert_changed, assert_same


# TODO: add tests that check that the MLP works as expected.
def test_MLP():
    pass


@pytest.mark.parametrize(
    "predictive_factors, num_users, num_items",
    [(5, 10, 10), (5, 10, 10), (5, 3, 10), (1, 3, 3)],
)
def test_output_shapes_NeuMFMLPModule(predictive_factors, num_users, num_items):
    """Check that no mather the inner settings of the network, the output is always correct"""
    mod = NeuMFMLPModule(predictive_factors, num_users, num_items, 0)

    user_tensor = torch.LongTensor([1, 2])
    item_tensor = torch.LongTensor([1, 2])

    res = mod(user_tensor, item_tensor)  # predict scores for items given the users

    assert res.shape == (2, 1)

    assert (res.detach().numpy() <= 1).all()
    assert (res.detach().numpy() >= 0).all()


def test_training_epoch(mat):
    X = mat.binary_values
    a = NeuMFMLPOnly(
        predictive_factors=4,
        n_negatives_per_positive=2,
    )
    device = a.device
    a._init_model(X)

    # Each training epoch should update the parameters
    params = [np for np in a.model_.named_parameters() if np[1].requires_grad]
    params_before = [(name, p.clone()) for (name, p) in params]
    for _ in range(5):
        a._train_epoch(X)
    params = [np for np in a.model_.named_parameters() if np[1].requires_grad]
    assert_changed(params_before, params, device)


@pytest.mark.parametrize("users", [[0, 1], [0], [0, 1, 3]])
def test_batch_predict(mat, users):
    a = NeuMFMLPOnly(
        predictive_factors=2,
        n_negatives_per_positive=2,
    )
    device = a.device
    a.fit(mat, (mat, mat))
    params = [np for np in a.model_.named_parameters() if np[1].requires_grad]
    params_before = [(name, p.clone()) for (name, p) in params]

    pred = a._batch_predict(mat.users_in(users), users=users)

    assert pred.shape == mat.shape
    np.testing.assert_array_equal(pred.sum(axis=1).nonzero()[0], users)

    params = [np for np in a.model_.named_parameters() if np[1].requires_grad]
    assert_same(params_before, params, device)


@pytest.mark.parametrize(
    "users, negatives",
    [
        (torch.LongTensor([4, 5, 6]), torch.LongTensor([[1, 2], [1, 2], [1, 2]])),
        (torch.LongTensor([4, 5, 6]), torch.LongTensor([[1], [1], [1]])),
    ],
)
def test_negative_input_construction(users, negatives):
    n_negatives_per_positive = negatives.shape[1]
    a = NeuMFMLPOnly(
        predictive_factors=4,
        n_negatives_per_positive=n_negatives_per_positive,
    )

    users_input, negatives_input = a._construct_negative_prediction_input(users, negatives)
    assert users_input.shape == negatives_input.shape
    assert len(users_input.shape) == 1  # 1d vectors

    # Check that both are in the right order (each user is repeated n_negatives_per_positive times before the next user is present)
    for ix in range(users_input.shape[0]):
        assert users_input[ix] == users[ix // n_negatives_per_positive]
        assert negatives_input[ix] == negatives[ix // n_negatives_per_positive, ix % n_negatives_per_positive]


def test_overfit(mat):
    m = NeuMFMLPOnly(
        predictive_factors=5,
        batch_size=1,
        max_epochs=20,
        learning_rate=0.02,
        stopping_criterion="ndcg",
        n_negatives_per_positive=1,
    )

    # set sampler to exact sampling
    m.sampler.exact = True
    m.fit(mat, (mat, mat))
    bin_mat = mat.binary_values
    pred = m.predict(mat.binary_values).toarray()
    for user in mat.active_users:
        # The model should have overfitted, so that the visited items have the highest similarities
        positives = bin_mat[user].nonzero()[1]
        negatives = list(set(range(mat.shape[1])) - set(positives))

        for item in positives:
            assert (pred[user][negatives] < pred[user, item]).all()
