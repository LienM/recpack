"""Code loosely based on DaisyRec implementation
at https://github.com/AmazingDD/daisyRec/
"""
from recpack.algorithms.base import Algorithm
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.sparse import csr_matrix, coo_matrix
import logging
import numpy as np

logger = logging.getLogger("recpack")


# TODO: Move to util file
def int_to_tensor(x: int):
    return torch.LongTensor([x])


# TODO: compare with VAE base class,
# and see if it makes sense to combine into a different inheritance hierarchy
# TODO: early stop parameter and use.
class BPRMF(Algorithm):
    """Implements Matrix Factorization by using the BPR-OPT objective
    and SGD optimization.

    The BPR optimization aims to construct a factorization that optimally
    ranks interesting items above uninteresting items for all users.

    :param num_components: The size of the latent vectors for both users and items.
                            defaults to 100
    :type num_components: int, optional
    :param reg: The regularization, determines with how much to
                regularize the parameter values, defaults to 0.0
    :type reg: float, optional
    :param num_epochs: The max amount of epochs to train the model, defaults to 20
    :type num_epochs: int, optional
    :param learning_rate: The learning rate of the optimization procedure,
                            defaults to 0.01
    :type learning_rate: float, optional
    :param weight_decay: The decay to be used in SGD, defaults to 0.0001
    :type weight_decay: float, optional
    :param seed: seed to fix random numbers, to make results reproducible,
                    defaults to None
    :type seed: [type], optional
    """

    def __init__(
        self,
        num_components=100,
        reg=0.0,
        num_epochs=20,
        learning_rate=0.01,
        weight_decay=0.0001,
        seed=None,
    ):

        self.num_components = num_components
        self.reg = reg
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.seed = seed
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        # placeholders, will get initialized in _init_model
        self.optimizer = None
        self.model_ = None
        self.steps = 0

    def _init_model(self, num_users, num_items):
        self.model_ = MFModule(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)

        self.optimizer = optim.SGD(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.steps = 0

    def _generate_samples(self, train_data: csr_matrix):
        # TODO: this is a silly naive implementation,
        # so please update to a smarter version :D

        negatives = np.ones(train_data.shape) - train_data
        users, items = train_data.nonzero()
        for u, i in zip(users, items):
            # sample y from negatives
            j = np.random.choice(negatives[u, :].nonzero()[1])
            yield (u, i, j)

    def _train_epoch(self, train_data: csr_matrix):
        train_loss = 0.0
        self.model_.train()

        # create random samples of user, item, item
        samples = self._generate_samples(train_data)
        for u, i, j in samples:
            # TODO: to tensor
            user = int_to_tensor(u).to(self.device)
            item_i = int_to_tensor(i).to(self.device)
            item_j = int_to_tensor(j).to(self.device)

            self.optimizer.zero_grad()
            i_sim = self.model_.forward(user, item_i)
            j_sim = self.model_.forward(user, item_j)
            # print(u, i, j, i_sim, j_sim)
            loss = self._compute_loss(i_sim, j_sim)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.steps += 1
        return train_loss

    def _compute_loss(self, i_sim, j_sim):
        """Computes BPR loss

        :param i_sim: [description]
        :type i_sim: [type]
        :param j_sim: [description]
        :type j_sim: [type]
        :return: [description]
        :rtype: [type]
        """

        # .sum makes this also usable for lists of similarities.
        bpr_loss = -(i_sim - j_sim).sigmoid().log().sum()

        #  Add regularization
        return bpr_loss + self.reg * (
            self.model_.item_embedding.weight.norm()
            + self.model_.user_embedding.weight.norm()
        )

    # TODO: check if validation data is necessary?
    def fit(self, X: csr_matrix):

        self._init_model(X.shape[0], X.shape[1])

        last_loss = 0.0
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(X)
            logger.info(f"training loss = {train_loss}")

            # TODO: early stopping
            # delta_loss = float(train_loss - last_loss)
            # if (abs(delta_loss) < 1e-5) and self.early_stop:
            #     print('Satisfy early stop mechanism')
            #     break

            last_loss = train_loss

        logger.info(f"finished training, final loss = {last_loss}")

    def predict(self, X: csr_matrix):
        # TODO: this is super naive, because the X is only used to find users.
        users = list(set(X.nonzero()[0]))
        mult_M = (
            self.model_.user_embedding.weight[users]
            @ self.model_.item_embedding.weight.T
        )

        result = np.zeros(X.shape)
        result[users] = mult_M.detach().cpu().numpy()
        return coo_matrix(result).tocsr()


# TODO: in a recommendation context, component is maybe not so clear,
# and all MF algorithms should use num_factors?
class MFModule(nn.Module):
    def __init__(self, num_users, num_items, num_components=100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, num_components)
        self.item_embedding = nn.Embedding(num_items, num_components)

        # Initialise embeddings to a random start
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, u, i):
        """Generate similarity u and i

        :param u: [description]
        :type u: [type]
        :param i: [description]
        :type i: [type]
        """
        ue = self.user_embedding(u)
        ie = self.item_embedding(i)

        return (ue * ie).sum(dim=-1)
