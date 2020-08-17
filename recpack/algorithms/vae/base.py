from typing import List, Tuple

import torch.nn as nn
import torch
from math import ceil
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from sklearn.utils.validation import check_is_fitted

import logging

from recpack.algorithms.base import Algorithm

from recpack.algorithms.vae.util import (
    naive_sparse2tensor
)

logger = logging.getLogger('recpack')

#######
#  Helper functions for batch computation
#######


def get_users(data):
    return list(set(data.nonzero()[0]))


def get_batches(users, batch_size=1000):
    return [
        users[i * batch_size: min((i * batch_size) + batch_size, len(users))]
        for i in range(ceil(len(users) / batch_size))
    ]


class VAE(Algorithm):
    def __init__(
        self,
        batch_size,
        max_epochs,
        seed,
        learning_rate,
        stopping_criterion,
        drop_negative_recommendations=True,
    ):
        self.batch_size = (
            # TODO * torch.cuda.device_count() if cuda else batch_size
            batch_size
        )
        self.max_epochs = max_epochs
        self.seed = seed
        torch.manual_seed(seed)

        self.learning_rate = learning_rate

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        self.stopping_criterion = stopping_criterion

        # if drop_negative recommendations is true,
        # we remove all negative scores in the response.
        # We need to do this to avoid blowing up RAM usage
        # converting the dense recommendation response to sparse
        self.drop_negative_recommendations = drop_negative_recommendations

    #######
    # FUNCTIONS TO OVERWRITE FOR EACH VAE
    #######

    def _init_model(self, dim_input_layer: int) -> None:
        """Initialize self.model_ based on the size of the input.
        Also initialize the optimizer(s)

        At the end of this function,
        the model used in this VAE should be initialized
        and the optimizers used to tune the parameters are initialized.

        :param dim_input_layer: the number of features in the input layer.
        :type dim_input_layer: int
        """

        self.model_ = None
        raise NotImplementedError()

    def _train_epoch(self, train_data: csr_matrix, users: List[int]):
        """Perform one training epoch.

        Overwrite this function with the concrete implementation
        of the training step.

        :param train_data: Training data (UxI)
        :type train_data: [type]
        :param users: List of all users in the training dataset.
        :type users: List[int]
        """
        # Set to training
        self.model_.train()
        raise NotImplementedError()

    def _compute_loss(self, X: torch.Tensor, X_pred: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute the prediction loss.

        Function used in training.
        Overwrite this function with the right way to compute the loss

        :param X: input data
        :type X: torch.Tensor
        :param X_pred: output data
        :type X_pred: torch.Tensor
        :param mu: the mean tensor
        :type mu: torch.Tensor
        :param logvar: the variance tensor
        :type logvar: torch.Tensor
        :return: the loss tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError()

    #######
    # STANDARD FUNCTIONS
    # DO NOT OVERWRITE UNLESS ABSOLUTELY NECESSARY
    #######
    def fit(self, X: csr_matrix,
            validation_data: Tuple[csr_matrix, csr_matrix]) -> None:
        """Fit the model on the interaction matrix,
        validation of the parameters is done with the validation data tuple.

        At the end of this function, the self.model_ should be ready for use.
        There is typically no need to change this function when
        inheritting from the base class.

        :param X: user interactions to train on
        :type X: csr_matrix
        :param validation_data: the data to use for validation,
                                the first entry in the tuple is the input data,
                                the second the expected output
        :type validation_data: Tuple[csr_matrix, csr_matrix]
        """

        self._init_model(X.shape[1])

        # TODO Investigate Multi-GPU
        # multi_gpu = False
        # if torch.cuda.device_count() > 1:
        #     self.model_ = torch.nn.DataParallel(self.model_)
        #     multi_gpu = True

        val_in, val_out = validation_data

        train_users = list(set(X.nonzero()[0]))
        val_users = list(set(val_in.nonzero()[0]))

        for epoch in range(0, self.max_epochs):
            self._train_epoch(X, train_users)
            self._evaluate(val_in, val_out, val_users)

        # Load best model, not necessarily last model
        self.load(self.stopping_criterion.best_value)

        return

    def _batch_predict(self, X: csr_matrix) -> csr_matrix:
        """Helper function to batch the prediction,
        to avoid going out of RAM on the GPU

        Will batch the nonzero users into batches of self.batch_size.

        :param X: The input user interaction matrix
        :type X: csr_matrix
        :return: The predicted affinity of users for items.
        :rtype: csr_matrix
        """
        results = np.zeros(X.shape)
        for batch in get_batches(get_users(X), batch_size=self.batch_size):
            in_ = X[batch]
            in_tensor = naive_sparse2tensor(in_).to(self.device)

            out_tensor, _, _ = self.model_(in_tensor)
            results[batch] = out_tensor.detach().cpu().numpy()

        logger.info(f"shape of response ({results.shape})")

        if self.drop_negative_recommendations:
            results[results < 0] = 0

        return coo_matrix(results).tocsr()

    def _batch_predict_and_loss(
            self, X: csr_matrix) -> Tuple[csr_matrix, torch.Tensor]:
        """Helper function to batch the prediction and loss computation,
        to avoid going out of RAM on the GPU.

        Will batch the nonzero users into batches of self.batch_size.
        The loss is accumulated by taking the sum.

        :param X: The input user interaction matrix
        :type X: csr_matrix
        :return: The predicted affinity of users for items
                 and the accumulated loss.
        :rtype: Tuple[csr_matrix, torch.Tensor]
        """

        results = np.zeros(X.shape)
        loss = 0
        for batch in get_batches(get_users(X), batch_size=self.batch_size):
            in_ = X[batch]
            in_tensor = naive_sparse2tensor(in_).to(self.device)

            out_tensor, mu, logvar = self.model_(in_tensor)
            loss += self._compute_loss(in_tensor, out_tensor, mu, logvar)

            results[batch] = out_tensor.detach().cpu().numpy()

        if self.drop_negative_recommendations:
            results[results < 0] = 0

        return csr_matrix(results), loss

    def _evaluate(self, val_in: csr_matrix,
                  val_out: csr_matrix, users: List[int]):
        # Set to evaluation
        self.model_.eval()

        with torch.no_grad():
            # Evaluate batched
            X_pred_cpu, loss = self._batch_predict_and_loss(val_in)
            val_loss = loss.item()
            X_true = val_out

            self.stopping_criterion.calculate(X_true, X_pred_cpu)
            logger.info(
                f"Evaluation Loss = {val_loss}"
                f", NDCG@100 = {self.stopping_criterion.value}"
            )
            if self.stopping_criterion.is_best:
                logger.info("Model improved. Storing better model.")
                self.save()
            self.stopping_criterion.reset()

    def load(self, value):
        # TODO Give better names
        with open(f"{self.name}_ndcg_100_{value}.trch", "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        with open(
            f"{self.name}_ndcg_100_{self.stopping_criterion.value}.trch", "wb"
        ) as f:
            torch.save(self.model_, f)

    def predict(self, X, user_ids=None):
        check_is_fitted(self)
        self.model_.eval()

        # Compute the result in batches.
        X_pred = self._batch_predict(X)

        return X_pred


class VAETorch(nn.Module):
    """
    Base class for building torch modules.
    """

    def __init__(
        self
    ):
        super().__init__()

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass the input through the network, and return result.

        :param x: input tensor
        :type x: torch.Tensor
        :return: A tuple with
                (predicted output value, mean values, average values)
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
