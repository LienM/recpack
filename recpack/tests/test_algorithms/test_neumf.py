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
    "num_components, num_users, num_items, hidden_sizes",
    [(5, 10, 10, [10]), (5, 10, 10, [10, 5, 3]), (5, 3, 10, [10]), (1, 3, 3, [10])],
)
def test_output_shapes_NeuMFMLPModule(num_components, num_users, num_items, hidden_sizes):
    """Check that no mather the inner settings of the network, the output is always correct"""
    mod = NeuMFMLPModule(num_components, num_users, num_items, hidden_sizes)

    user_tensor = torch.LongTensor([1, 2])
    item_tensor = torch.LongTensor([1, 2])

    res = mod(user_tensor, item_tensor)  # predict scores for items given the users

    assert res.shape == (2, 1)

    assert (res.detach().numpy() <= 1).all()
    assert (res.detach().numpy() >= 0).all()


def test_training_epoch(mat):
    X = mat.binary_values
    a = NeuMFMLPOnly(
        num_components=3,
        hidden_dims=[6, 4],
        batch_size=2,
        max_epochs=50,
        learning_rate=0.005,
        stopping_criterion="ndcg",
        U=2,
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
        num_components=3,
        hidden_dims=[6, 4],
        batch_size=2,
        max_epochs=10,
        learning_rate=0.005,
        stopping_criterion="ndcg",
        U=2,
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
    U = negatives.shape[1]
    a = NeuMFMLPOnly(
        num_components=3,
        hidden_dims=[16, 8, 4],
        batch_size=100,
        max_epochs=10,
        learning_rate=0.01,
        stopping_criterion="ndcg",
        U=U,
    )

    users_input, negatives_input = a._construct_negative_prediction_input(users, negatives)
    assert users_input.shape == negatives_input.shape
    assert len(users_input.shape) == 1  # 1d vectors

    # Check that both are in the right order (each user is repeated U times before the next user is present)
    for ix in range(users_input.shape[0]):
        assert users_input[ix] == users[ix // U]
        assert negatives_input[ix] == negatives[ix // U, ix % U]
