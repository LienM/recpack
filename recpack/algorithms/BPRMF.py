import logging

from scipy.sparse import csr_matrix, lil_matrix

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import bpr_loss
from recpack.algorithms.samplers import bootstrap_sample_pairs
from recpack.algorithms.stopping_criterion import (
    StoppingCriterion,
)
from recpack.algorithms.util import (
    get_users,
    get_batches,
)

logger = logging.getLogger("recpack")


class BPRMF(TorchMLAlgorithm):
    """Implements Matrix Factorization by using the BPR-OPT objective
    and SGD optimization.

    MF implementation using the BPR criterion as defined in Rendle, Steffen, et al.
    "BPR: Bayesian personalized ranking from implicit feedback."

    The BPR optimization criterion aims to construct a factorization that optimally
    ranks interesting items (interacted with previously)
    above uninteresting or unknown items for all users.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import BPRMF

        # Since BPRMF uses iterative optimisation, it needs validation data
        # To decide which of the iterations yielded the best model
        # This validation data should be split into an input and output matrix.
        # In this example the data has been split in a strong generalization fashion
        X = csr_matrix(np.array(
            [[1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]])
        )
        x_val_in = csr_matrix(np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
        )
        x_val_out = csr_matrix(np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]])
        )

        algo = BPRMF(num_components=5, batch_size=3, max_epochs=4)
        # Fit algorithm
        algo.fit(X, (x_val_in, x_val_out))

        # Recommend for the validation input data,
        # so we can inspect what the model learned
        # In a realistic setting you would have a test dataset
        predictions = algo.predict(x_val_in)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()


    :param num_components: The size of the latent vectors for both users and items.
                            defaults to 100
    :type num_components: int, optional
    :param lambda_h: the regularization parameter for the item embedding,
        should be a value between 0 and 1.
        Defaults to 0.0
    :type lambda_h: float, optional
    :param lambda_w: the regularization parameter for the user embedding,
        defaults to 0.0
    :type lambda_w: float, optional
    :param batch_size: size of the batches to use during gradient descent. Defaults to 1000.
    :type batch_size: int, optional
    :param max_epochs: The max amount of epochs to train the model, defaults to 20
    :type max_epochs: int, optional
    :param learning_rate: The learning rate of the optimization procedure,
                            defaults to 0.01
    :type learning_rate: float, optional
    :param seed: seed to fix random numbers, to make results reproducible,
                    defaults to None
    :type seed: int, optional,
    :param stopping_criterion: Which criterion to use optimise the parameters,
        a string which indicates the name of the stopping criterion.
        Which criterions are available can be found at
        recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS.
        Defaults to 'recall'
    :type stopping_criterion: str, optional
    :param stop_early: If True, early stopping is enabled,
        and after 5 iterations where improvement of loss function
        is below 0.01 the optimisation is stopped, even if max_epochs is not reached.
        Defaults to False
    :type stop_early: bool, optional
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
    """

    def __init__(
        self,
        num_components=100,
        lambda_h=0.0,
        lambda_w=0.0,
        batch_size=1_000,
        max_epochs=20,
        learning_rate=0.01,
        stopping_criterion: str = "bpr",
        stop_early: bool = False,
        seed=None,
        save_best_to_file: bool = False,
    ):

        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            StoppingCriterion.create(stopping_criterion, stop_early=stop_early),
            seed,
            save_best_to_file,
        )

        self.num_components = num_components
        self.lambda_h = lambda_h
        self.lambda_w = lambda_w
        self.learning_rate = learning_rate

        self.stop_early = stop_early

    def _init_model(self, X: csr_matrix):
        num_users, num_items = X.shape
        self.model_ = MFModule(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)

        self.optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate)
        self.steps = 0

    def _batch_predict(self, X):
        """Helper function for predict, so we can also use it in validation loss
        without the model being fitted"""

        results = lil_matrix(X.shape)
        for users in get_batches(get_users(X), batch_size=self.batch_size):

            user_tensor = torch.LongTensor(users).to(self.device)
            item_tensor = torch.arange(X.shape[1]).to(self.device)

            results[users] = (
                self.model_(user_tensor, item_tensor).detach().cpu().numpy()
            )

        return results.tocsr()

    def _train_epoch(self, train_data: csr_matrix):
        """train a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
        :type train_data: csr_matrix
        """
        losses = []
        self.model_.train()

        for users, target_items, mnar_items in tqdm(
            bootstrap_sample_pairs(
                train_data, batch_size=self.batch_size, sample_size=train_data.nnz
            ),
            desc="train_epoch BPRMF",
        ):
            users = users.to(self.device)
            # Target items are items the user has interacted with,
            # and we expect to recommend high
            target_items = target_items.to(self.device)
            # Items the user has not seen, and assuming MNAR data
            mnar_items = mnar_items.to(self.device)

            self.optimizer.zero_grad()

            target_sim = self.model_.forward(users, target_items).diag()
            mnar_sim = self.model_.forward(users, mnar_items).diag()

            # Checks to make sure the shapes are correct.
            assert mnar_sim.shape == target_sim.shape
            assert target_sim.shape[0] == users.shape[0]

            loss = self._compute_loss(target_sim, mnar_sim)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            self.steps += 1

    def _compute_loss(self, positive_sim, negative_sim):

        loss = bpr_loss(positive_sim, negative_sim)
        loss += (
            self.lambda_h * self.model_.item_embedding.weight.norm()
            + self.lambda_w * self.model_.user_embedding.weight.norm()
        )

        return loss


class MFModule(nn.Module):
    """MF torch module, encodes the embeddings and the forward functionality.

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param num_components: The size of the embedding per user and item, defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_users, num_items, num_components=100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, num_components)  # User embedding
        self.item_embedding = nn.Embedding(num_items, num_components)  # Item embedding

        self.std = 1 / num_components ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.user_embedding.weight, std=self.std)
        nn.init.normal_(self.item_embedding.weight, std=self.std)

    def forward(
        self, user_tensor: torch.Tensor, item_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dot-product of user embedding (w_u) and item embedding (h_i)
        for every user and item pair in user_tensor and item_tensor.

        :param user_tensor: [description]
        :type user_tensor: [type]
        :param item_tensor: [description]
        :type item_tensor: [type]
        """
        w_u = self.user_embedding(user_tensor)
        h_i = self.item_embedding(item_tensor)

        return w_u.matmul(h_i.T)
