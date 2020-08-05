import time
from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.validation import check_is_fitted

import logging

from recpack.splitters.splitter_base import batch
from recpack.algorithms.algorithm_base import Algorithm

from recpack.algorithms.torch_algorithms.utils import naive_sparse2tensor, StoppingCriterion

logger = logging.getLogger('recpack')

class VAE(Algorithm):
    def __init__(
        self,
        batch_size,
        max_epochs,
        seed,
        learning_rate
    ):
        self.batch_size = (
            batch_size  # TODO * torch.cuda.device_count() if cuda else batch_size
        )
        self.max_epochs = max_epochs
        self.seed = seed
        torch.manual_seed(seed)

        self.learning_rate = learning_rate

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        self.stopping_criterion = StoppingCriterion(100)

    #######
    # FUNCTIONS TO OVERWRITE FOR EACH VAE
    #######
 
    def _init_model(self, dim_input_layer: int) -> None:
        """Initialize self.model_ based on the size of the input.
        Also initialize the optimizer(s)

        At the end of this function, the model used in this VAE should be initialized
        and the optimizers used to tune the parameters are initialized.

        :param dim_input_layer: the number of features in the input layer.
        :type dim_input_layer: int
        """

        self.model_ = None
        raise NotImplementedError()
    
    def _train_epoch(self, train_data: csr_matrix, users: List[int]):
        """Perform one training epoch.

        Overwrite this function with the concrete implementation of the training step.

        :param train_data: Training data (UxI)
        :type train_data: [type]
        :param users: List of all users in the training dataset.
        :type users: List[int]
        """
        # Set to training
        self.model_.train()
        raise NotImplementedError()

    def _compute_pred(self, val_X: torch.Tensor) -> torch.Tensor:
        """Compute just the predicted output.

        Used in the prediction step.
        Overwrite this function with the right way to compute the prediction


        :param val_X: input data
        :type val_X: torch.Tensor        
        :return: The recommendations as a tensor.
        :rtype: torch.Tensor
        """
        raise NotImplementedError()

    def _compute_pred_and_loss(self, val_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the prediction and the loss of that prediction.

        Function used in training.
        Overwrite this function with the right way to compute the loss and prediction

        :param val_X: input data
        :type val_X: torch.Tensor
        :return: a tuple with the first element the predictions as a tensor, the second element is the loss.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError()

    #######
    # STANDARD FUNCTIONS
    # DO NOT OVERWRITE UNLESS ABSOLUTELY NECESSARY
    #######
    def fit(self, X: csr_matrix, validation_data: Tuple[csr_matrix, csr_matrix]) -> None:
        """Fit the model on the interaction matrix, validation of the parameters is done with the validation data tuple.

        At the end of this function, the self.model_ should be ready for use.
        There is typically no need to change this function when inheritting from the base class.

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


    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix, users: List[int]):
        val_loss = 0.0
        # Set to evaluation
        self.model_.eval()

        with torch.no_grad():
            val_X = naive_sparse2tensor(val_in).to(self.device)

            val_X_pred, loss = self._compute_pred_and_loss(val_X)
            val_loss += loss.item()

            val_X_pred_cpu = csr_matrix(val_X_pred.cpu())
            val_X_true = val_out

            self.stopping_criterion.calculate(val_X_true, val_X_pred_cpu)

        logger.info(
            f"Evaluation Loss = {val_loss}, NDCG@100 = {self.stopping_criterion.value}"
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

        users = list(set(X.nonzero()[0]))

        tensorX = naive_sparse2tensor(X[users, :]).to(self.device)

        tensorX_pred = self._compute_pred(tensorX)

        # [[0, 1, 2], [3, 4, 5]] -> [0, 1, 2, 3, 4, 5]
        V = tensorX_pred.cpu().flatten().detach().numpy()  # Flattens row-major.
        # -> [1, 2] -> 1, 1, 1, 2, 2, 2 (2 users x 3 items)
        U = np.repeat(users, X.shape[1])
        # -> [1, 2, 3] -> 1, 2, 3, 1, 2, 3 (2 users x 3 items)
        I = list(range(0, X.shape[1])) * len(users)

        X_pred = csr_matrix((V, (U, I)), shape=X.shape)

        return X_pred

