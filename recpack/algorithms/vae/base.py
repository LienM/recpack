import logging
import tempfile
from typing import List, Tuple

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils.validation import check_is_fitted
import torch

from recpack.algorithms.base import Algorithm
from recpack.algorithms.util import (
    naive_sparse2tensor,
    get_batches,
    get_users
)
from recpack.algorithms.stopping_criterion import StoppingCriterion, EarlyStoppingException
from recpack.data.matrix import Matrix, to_csr_matrix

logger = logging.getLogger("recpack")

"""
TODO Create one base Torch algorithm.
Things in common between all Torch/iterative algorithms:
- Stopping Criterion
- Train epoch method
- evaluate method
- fit/predict
- loss function
"""


class VAE(Algorithm):
    def __init__(
        self,
        batch_size,
        max_epochs,
        seed,
        learning_rate,
        stopping_criterion: StoppingCriterion,
        drop_negative_recommendations=True,
        save_best_to_file=False,
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

        self.best_model = tempfile.NamedTemporaryFile()
        self.save_best_to_file = save_best_to_file

    def __del__(self):
        """cleans up temp file"""
        self.best_model.close()

    #######
    # FUNCTIONS TO OVERWRITE FOR EACH VAE
    #######

    def _init_model(self, dim_input_layer: int) -> None:
        """
        Initialize self.model_ based on the size of the input.
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
        """
        Perform one training epoch.

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

    def _compute_loss(
        self,
        X: torch.Tensor,
        X_pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the prediction loss.

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
    def fit(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]) -> None:
        """
        Fit the model on the interaction matrix,
        validation of the parameters is done with the validation data tuple.

        At the end of this function, the self.model_ should be ready for use.
        There is typically no need to change this function when
        inheriting from the base class.

        :param X: user interactions to train on
        :type X: Matrix
        :param validation_data: the data to use for validation,
                                the first entry in the tuple is the input data,
                                the second the expected output
        :type validation_data: Tuple[Matrix, Matrix]
        """
        X, validation_data = to_csr_matrix((X, validation_data), binary=True)

        self._init_model(X.shape[1])

        # TODO Investigate Multi-GPU
        # multi_gpu = False
        # if torch.cuda.device_count() > 1:
        #     self.model_ = torch.nn.DataParallel(self.model_)
        #     multi_gpu = True

        val_in, val_out = validation_data

        try:
            for epoch in range(0, self.max_epochs):
                self._train_epoch(X)
                self._evaluate(val_in, val_out)
        except EarlyStoppingException:
            pass

        # Load best model, not necessarily last model
        self._load_best()

        if self.save_best_to_file:
            self.save()

        return

    def _batch_predict(self, X: csr_matrix) -> csr_matrix:
        """
        Helper function to batch the prediction,
        to avoid going out of RAM on the GPU

        Will batch the nonzero users into batches of self.batch_size.

        :param X: The input user interaction matrix
        :type X: csr_matrix
        :return: The predicted affinity of users for items.
        :rtype: csr_matrix
        """
        results = lil_matrix(X.shape)
        for batch in get_batches(get_users(X), batch_size=self.batch_size):
            in_ = X[batch]
            in_tensor = naive_sparse2tensor(in_).to(self.device)

            out_tensor, _, _ = self.model_(in_tensor)
            batch_results = out_tensor.detach().cpu().numpy()

            if self.drop_negative_recommendations:
                batch_results[batch_results < 0] = 0

            results[batch] = batch_results

        logger.info(f"shape of response ({results.shape})")

        return results.tocsr()

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix):
        # Set to evaluation
        self.model_.eval()

        with torch.no_grad():
            # Evaluate batched
            X_pred_cpu = self._batch_predict(val_in)
            X_true = val_out

            better = self.stopping_criterion.update(X_true, X_pred_cpu)

            if better:
                logger.info("Model improved. Storing better model.")
                self._save_best()

    @property
    def filename(self):
        # TODO: validation func used in name
        return f"{self.name}_loss_{self.stopping_criterion.best_value}.trch"

    def load(self, filename):
        with open(filename, "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        """Save the current model to disk"""
        with open(self.filename, "wb") as f:
            torch.save(self.model_, f)

    def _save_best(self):
        """Save the best model in a temp file"""
        self.best_model.close()
        self.best_model = tempfile.NamedTemporaryFile()
        torch.save(self.model_, self.best_model)

    def _load_best(self):
        self.best_model.seek(0)
        self.model_ = torch.load(self.best_model)

    def predict(self, X: Matrix):
        check_is_fitted(self)

        X = to_csr_matrix(X, binary=True)

        self.model_.eval()

        # Compute the result in batches.
        X_pred = self._batch_predict(X)

        self._check_prediction(X_pred, X)

        return X_pred
