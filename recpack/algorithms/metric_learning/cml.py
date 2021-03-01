from copy import deepcopy
import logging
import tempfile
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import Algorithm
from recpack.algorithms.samplers import warp_sample_pairs
from recpack.algorithms.loss_functions import covariance_loss, warp_loss
from recpack.algorithms.stopping_criterion import StoppingCriterion, EarlyStoppingException
from recpack.algorithms.util import sample

from recpack.data.matrix import Matrix, to_csr_matrix


logger = logging.getLogger("recpack")


"""
- Train before predict => Implement and write test
- Keep last => Implement and write test (How should I test this?)
- Stopping Criterion WARP => Fill in kwargs?
- Stopping Criterion: Just realised you can't actually pass stop_early etc. anymore the way we've set it up now...
- Add documentation

- Refactor predict in a cleaner way. E.g. the PyTorch model has an "export to annoy" function
that can be used to export a model that can be used for prediction, that has no entries for
users it doesn't know.
"""


class CML(Algorithm):
    def __init__(
        self,
        num_components: int,
        margin: float,
        learning_rate: float,
        num_epochs: int,
        seed: int = 42,
        batch_size: int = 50000,
        U: int = 20,
        stopping_criterion: str = "recall",
        save_best_to_file=True,
        disentangle=False,
        train_before_predict=False,
        keep_last=False
    ):
        """Collaborative Metric Learning.

        Pytorch Implementation of
        Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
        http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf without features, referred to as CML in the paper.

        :param num_components: Embedding dimension
        :type num_components: int
        :param margin: Hinge loss margin. Required difference in score, smaller than which we will consider a negative sample item in violation
        :type margin: float
        :param learning_rate: Learning rate for AdaGrad optimization
        :type learning_rate: float
        :param num_epochs: Number of epochs
        :type num_epochs: int
        :param seed: Random seed used for initialization, defaults to 42
        :type seed: int, optional
        :param batch_size: Sample batch size, defaults to 50000
        :type batch_size: int, optional
        :param U: Number of negative samples used in WARP loss function for every positive sample, defaults to 20
        :type U: int, optional
        :param stopping_criterion: Used to identify the best model computed thus far.
            The string indicates the name of the stopping criterion.
            Which criterions are available can be found at StoppingCriterion.FUNCTIONS
            Defaults to 'recall'
        :type stopping_criterion: str, optional
        :param save_best_to_file: If True, the best model is saved to disk after fit, defaults to True.
        :type save_best_to_file: bool, optional
        :param disentangle: Disentangle embedding dimensions by adding a covariance loss term to regularize, defaults to False
        :type disentangle: bool, optional
        :param train_before_predict: Compute user embeddings for unknown users in `predict`, defaults to False
        :type train_before_predict: bool, optional
        :param keep_last: Retain last model, rather than best (according to stopping criterion value on validation data), defaults to False
        :type keep_last: bool, optional
        """
        self.num_components = num_components
        self.margin = margin
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.U = U
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        self.save_best_to_file = save_best_to_file
        self.stopping_criterion = StoppingCriterion.create(stopping_criterion)
        self.disentangle = disentangle
        self.train_before_predict = train_before_predict
        self.keep_last = keep_last

    def _init_model(self, X):
        """Initialize model.

        :param num_users: Number of users.
        :type num_users: int
        :param num_items: Number of items.
        :type num_items: int
        """
        # Run only once. If a model already exists, skip
        if not self.model_:
            num_users, num_items = X.shape
            self.model_ = CMLTorch(
                num_users, num_items, num_components=self.num_components
            ).to(self.device)

            self.optimizer = optim.Adagrad(
                self.model_.parameters(), lr=self.learning_rate)

            self.known_users_ = set(X.nonzero()[0])

            self.best_model_ = tempfile.NamedTemporaryFile()
            torch.save(self.model_, self.best_model_)

    @property
    def filename(self) -> str:
        return f"{self.name}_loss_{self.stopping_criterion.best_value}.trch"

    # TODO: loading just the model is not enough to reuse it.
    # We also need known users etc. Pickling seems like a good way to go here.
    def load(self, filename: str) -> None:
        """Load a previously computed model.

        :param filename: path to file to load
        :type filename: str
        """
        with open(filename, "rb") as f:
            self.model_ = torch.load(f)

    def save(self) -> None:
        """Save the current model to disk"""
        with open(self.filename, "wb") as f:
            torch.save(self.model_, f)

    def _save_best(self) -> None:
        """Save the best model in a temp file"""
        # TODO Rename to save_temp
        # First removes the old saved file,
        # and then creates a new one in which the model is saved
        self.best_model_.close()
        self.best_model_ = tempfile.NamedTemporaryFile()
        torch.save(self.model_, self.best_model_)

    def _load_best(self) -> None:
        self.best_model_.seek(0)
        self.model_ = torch.load(self.best_model_)

    def fit(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]):
        """Fit the model on the X dataset, and evaluate model quality on validation_data

        :param X: The training data matrix
        :type X: Matrix
        :param validation_data: Validation data, as matrix to be used as input and matrix to be used as output
        :type validation_data: Tuple[csr_matrix, csr_matrix]
        """
        X, validation_data = to_csr_matrix((X, validation_data), binary=True)

        self._init_model(X)
        try:
            for epoch in range(self.num_epochs):
                self._train_epoch(X, self.model_, self.optimizer)
                self._evaluate(validation_data)
        except EarlyStoppingException:
            pass

        # Load the best of the models during training.
        if not self.keep_last:
            self._load_best()

        # If saving turned on: save to file.
        if self.save_best_to_file:
            self.save()

        return self

    def _batch_predict(
        self,
        users_to_predict_for: set,
        UEmb_as_tensor: torch.Tensor,
        IEmb_as_tensor: torch.Tensor,
    ) -> csr_matrix:
        """
        # TODO If approximate users doesn't work, this method need not be so complex
        Method for internal use only. Users should use `predict`.

        :param users_to_predict_for: Set of users for which we wish to make predictions
        :type users_to_predict_for: set
        :param UEmb_as_tensor: User embedding matrix UEmb
        :type UEmb_as_tensor: torch.Tensor
        :param IEmb_as_tensor: Item embedding matrix IEmb
        :type IEmb_as_tensor: torch.Tensor
        :raises: AssertionError when the input and model's number of items and users are incongruent
        :return: Predictions for all users in users_to_predict_for. Matrix has shape (model.num_users, model.num_items)
        :rtype: csr_matrix
        """
        U = torch.LongTensor(list(users_to_predict_for)).repeat_interleave(
            self.model_.num_items
        )
        I = torch.arange(self.model_.num_items).repeat(
            len(users_to_predict_for))

        num_interactions = U.shape[0]

        V = np.array([])

        for batch_ix in range(0, num_interactions, 10000):
            batch_U = U[batch_ix: min(num_interactions, batch_ix + 10000)].to(
                self.device
            )
            batch_I = I[batch_ix: min(num_interactions, batch_ix + 10000)].to(
                self.device
            )

            batch_w_U = UEmb_as_tensor[batch_U]
            batch_h_I = IEmb_as_tensor[batch_I]

            # Score = -distance
            batch_V = (
                -nn.PairwiseDistance(p=2)(batch_w_U,
                                          batch_h_I).detach().cpu().numpy()
            )

            V = np.append(V, batch_V)

        X_pred = csr_matrix(
            (V, (U.numpy(), I.numpy())),
            shape=(self.model_.num_users, self.model_.num_items),
        )

        return X_pred

    # TODO Predict using approximate nearest neighbours in Annoy?
    def predict(self, X: Matrix) -> csr_matrix:
        """Predict recommendations for each user with at least a single event in their history.

        :param X: interaction matrix, should have same size as model.
        :type X: Matrix
        :raises an: AssertionError when the input and model's number of items and users are incongruent.
        :return: csr matrix of same shape, with recommendations.
        :rtype: csr_matrix
        """
        check_is_fitted(self)

        X = to_csr_matrix(X, binary=True)

        assert X.shape == (self.model_.num_users, self.model_.num_items)

        if self.train_before_predict:
            # Please forgive me for this terrible piece of code.
            stopping_criterion = StoppingCriterion.create("warp", stop_early=True)

            predict_model = deepcopy(self)
            optimizer = optim.Adagrad(
                predict_model.model_.UEmb.parameters(), lr=self.learning_rate)
            predict_model.optimizer = optimizer
            predict_model.stopping_criterion = stopping_criterion

            predict_model.fit(X)

            # Extract embedding matrices
            UEmb_as_tensor = predict_model.model_.UEmb.state_dict()["weight"]
            IEmb_as_tensor = predict_model.model_.IEmb.state_dict()["weight"]
        else:

            # Extract embedding matrices
            UEmb_as_tensor = self.model_.UEmb.state_dict()["weight"]
            IEmb_as_tensor = self.model_.IEmb.state_dict()["weight"]

        users = set(X.nonzero()[0])

        X_pred = self._batch_predict(users, UEmb_as_tensor, IEmb_as_tensor)

        self._check_prediction(X_pred, X)

        return X_pred

    def _train_epoch(self, train_data: csr_matrix):
        """
        Train model for a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix
        :type train_data: csr_matrix
        """
        losses = []
        self.model_.train()

        for users, positives_batch, negatives_batch in tqdm(
            warp_sample_pairs(train_data, U=self.U, batch_size=self.batch_size)
        ):
            users = users.to(self.device)
            positives_batch = positives_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            self.optimizer.zero_grad()

            current_batch_size = users.shape[0]

            dist_pos_interaction = self.model_.forward(users, positives_batch)
            dist_neg_interaction_flat = self.model_.forward(
                users.repeat_interleave(self.U),
                negatives_batch.reshape(
                    current_batch_size * self.U, 1).squeeze(-1),
            )
            dist_neg_interaction = dist_neg_interaction_flat.reshape(
                current_batch_size, -1
            )

            loss = self._compute_loss(
                dist_pos_interaction.unsqueeze(-1), dist_neg_interaction
            )
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        logger.info(f"training loss = {np.mean(losses)}")

    def _evaluate(self, validation_data: Tuple[csr_matrix, csr_matrix]):
        """
        Perform evaluation step, samples get drawn
        from the validation data, and compute loss.

        If loss improved over previous epoch, store the model, and update best value.

        :param validation_data: validation data interaction matrix
        :type validation_data: Tuple[csr_matrix, csr_matrix]
        """
        self.model_.eval()
        with torch.no_grad():
            # Need to make a selection, otherwise this step is way too slow.
            val_data_in, val_data_out = validation_data
            val_data_in_selection, val_data_out_selection = sample(
                val_data_in, val_data_out, sample_size=1000)

            # Extract embedding matrices
            UEmb_as_tensor = self.model_.UEmb.state_dict()["weight"]
            IEmb_as_tensor = self.model_.IEmb.state_dict()["weight"]

            validation_users = set(val_data_in_selection.nonzero()[0])

            X_val_pred = self._batch_predict(
                validation_users, UEmb_as_tensor, IEmb_as_tensor)
            X_val_pred[val_data_in_selection.nonzero()] = 0
            better = self.stopping_criterion.update(
                val_data_out_selection, X_val_pred)

            if better:
                self._save_best()

    def _compute_loss(
        self, dist_pos_interaction: torch.Tensor, dist_neg_interaction: torch.Tensor
    ) -> torch.Tensor:
        """
        Method for internal use only. Please use `warp_loss` or `covariance_loss`.

        Compute differentiable loss function.

        :param dist_pos_interaction: Tensor containing distances between positive sample pairs.
        :type dist_pos_interaction: torch.Tensor
        :param dist_neg_interaction: Tensor containing distance between negative sample pairs.
        :type dist_neg_interaction: torch.Tensor
        :return: 0-D Tensor containing computed loss value.
        :rtype: torch.Tensor
        """
        loss = warp_loss(
            dist_pos_interaction,
            dist_neg_interaction,
            self.margin,
            self.model_.num_items,
            self.U,
        )

        if self.disentangle:
            loss += covariance_loss(self.model_.IEmb, self.model_.UEmb)

        return loss


class CMLTorch(nn.Module):
    """
    Implementation of CML in PyTorch.

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param num_components: The size of the embedding per user and item, defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_users: int, num_items: int, num_components: int = 100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.UEmb = nn.Embedding(num_users, num_components)  # User embedding
        self.IEmb = nn.Embedding(num_items, num_components)  # Item embedding

        self.std = 1 / num_components ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.UEmb.weight, std=self.std)
        nn.init.normal_(self.IEmb.weight, std=self.std)

        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, U: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidian distance between user embedding (w_u) and item embedding (h_i)
        for every user and item pair in U and I.

        :param U: User identifiers.
        :type U: torch.Tensor
        :param I: Item identifiers.
        :type I: torch.Tensor
        :return: Euclidian distances between user-item pairs.
        :rtype: torch.Tensor
        """
        w_U = self.UEmb(U)
        h_I = self.IEmb(I)

        # U and I are unrolled -> [u=0, i=1, i=2, i=3] -> [0, 1], [0, 2], [0,3]

        return self.pdist(w_U, h_I)
