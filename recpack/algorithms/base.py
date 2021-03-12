import logging
import time
import warnings

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import tempfile
import torch
from typing import Tuple

from recpack.algorithms.stopping_criterion import (
    EarlyStoppingException,
    StoppingCriterion,
)
from recpack.algorithms.util import (
    naive_sparse2tensor,
    get_batches,
    get_users,
)
from recpack.data.matrix import to_csr_matrix, Matrix

logger = logging.getLogger("recpack")


class Algorithm(BaseEstimator):
    """Base class for all algorithm implementations.

    Each algorithm should have a fit and predict method.

    TODO? Add documentation how to create your own algorithm.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """name of the object, combined with the parameters.

        Constructed by recreating the initialisation call.
        Example: ``Algorithm(param_1=value)``
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def set_params(self, **params):
        """Set the parameters of the estimator.

        :param params: Estimator parameters
        :type params: dict
        """
        super().set_params(**params)

    def _fit(self, X: csr_matrix):
        raise NotImplementedError("Please implement _fit")

    def _predict(self, X: csr_matrix) -> csr_matrix:
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self):
        """Helper function to check that fit stored information."""
        # Use the sklear check_is_fitted function,
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        check_is_fitted(self)

    def _check_prediction(self, X_pred: csr_matrix, X: csr_matrix) -> None:
        """Checks that the prediction matches expectations.

        Checks implemented
        - Check that all users with history got at least 1 recommendation

        :param X_pred: The matrix with predictions
        :type X_pred: csr_matrix
        :param X: The input matrix for prediction, used as 'history'
        :type X: csr_matrix
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warnings.warn(
                f"{self.name} failed to recommend any items "
                f"for {len(missing)} users"
            )

    def fit(self, X: Matrix) -> "Algorithm":
        """Fit the model to the input interaction matrix.

        After fitting the model will be ready to use for prediction.

        :param X: The interactions to fit the model on.
        :type X: Matrix
        :return: **self**, fitted algorithm
        :rtype: Algorithm
        """
        start = time.time()
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"fitting {self.name} complete - Took {start - end :.3}s")
        return self

    def predict(self, X: Matrix) -> csr_matrix:
        """Predicts scores, given the interactions in X.

        Recommends items for each nonzero user in the X matrix.

        :param X: interactions to predict from.
        :type X: Matrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_matrix
        """
        check_is_fitted(self)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred


class ItemSimilarityMatrixAlgorithm(Algorithm):
    """Base algorithm for algorithms that fit an item to item similarity model.

    Fit will fill in the ``similarity_matrix_`` property,
    which encodes the similarity between items.
    This can be a dense or sparse matrix depending on the algorithms.

    Prediction will compute the dot product of the history vector of a user
    and the similarity matrix.
    """

    def _predict(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        scores = X @ self.similarity_matrix_

        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores

    def _check_fit_complete(self):
        """Checks if the fitted matrix, contains a similarity for each item.
        Uses warnings to push this info to the customer.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar items for {missing} items.")


class TopKItemSimilarityMatrixAlgorithm(ItemSimilarityMatrixAlgorithm):
    """Base class for algorithms where only the K most similar items are used for each item."""

    def __init__(self, K):
        super().__init__()
        self.K = K


class FactorizationAlgorithm(Algorithm):
    """Base class for Factorization algorithms.

    A factorization algorithm fits a user feature and item feature matrix.

    Prediction happens by multiplying a users features with the item features.

    TODO -> Add info for creating your own factorization algorithm
    TODO -> In the Neural Network we call things embeddings,
    probably should call the features here embeddings?

    :param num_components: the dimension of the feature matrices. defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_components=100):
        super().__init__()
        self.num_components = num_components

    def _predict(self, X: Matrix):
        assert X.shape == (self.user_features_.shape[0], self.item_features_.shape[1])
        users = list(set(X.nonzero()[0]))
        result = lil_matrix(X.shape)
        result[users] = self.user_features_[users] @ self.item_features_
        return result.tocsr()


class TorchMLAlgorithm(Algorithm):
    """Base class for algorithms using torch to learn a metric.

    Fits a neural network to optimise a loss function.
    During training a number of epochs is used, during each epoch
    the model is updated, and evaluated.

    During evaluation the stopping criterion is used to manage
    which is the best fitted model,
    and if we can stop training early if not enough improvement
    has been made in the last updates.

    After training the best model will be loaded, and used for subsequent prediction.

    The batch size is also used for prediction, to keep reduce load on GPU RAM.

    :param batch_size: How many samples to use in each update step.
        Higher batch sizes make each epoch more efficient,
        but increases the amount of epochs needed to converge to the optimum,
        by reducing the amount of updates per epoch.
    :type batch_size: int
    :param max_epochs: The max number of epochs to train,
        if the stopping criterion uses early stopping,
        it might stop earlier if the model on optimal weights.
    :type max_epochs: int
    :param learning_rate: How much to update the weights at each update.
    :type learning_rate: float
    :param stopping_criterion: The stopping criterion to use for training.
    :type stopping_criterion: StoppingCriterion
    :param seed: seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional

    """

    def __init__(
        self,
        batch_size,
        max_epochs,
        learning_rate,
        stopping_criterion: StoppingCriterion,
        seed=None,
        save_best_to_file=False,
    ):
        # TODO batch_size * torch.cuda.device_count() if cuda else batch_size
        #   -> Multi GPU
        self.batch_size = batch_size

        self.max_epochs = max_epochs

        self.learning_rate = learning_rate
        self.stopping_criterion = stopping_criterion

        self.seed = seed
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # TODO: base VAE had drop_negative_recommendations option to spare RAM usage
        # A better solution would be a prune style action like EASE
        #   -> keep highest absolute values

        self.save_best_to_file = save_best_to_file

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

    def _init_model(self, X: Matrix) -> None:
        """Initialise the torch model that will learn the weights.

        :param X: a user item interaction Matrix.
        :type X: Matrix
        """
        raise NotImplementedError()

    def _save_best(self):
        """Save the best model in a temp file"""
        self.best_model.close()
        self.best_model = tempfile.NamedTemporaryFile()
        torch.save(self.model_, self.best_model)

    def _load_best(self):
        self.best_model.seek(0)
        self.model_ = torch.load(self.best_model)

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix) -> None:
        """Perform evaluation step, should call self.stopping_criterion.update.

        :param val_in: Validation Data input
        :type val_in: csr_matrix
        :param val_out: Expected output from validation data
        :type val_out: csr_matrix
        """
        # Evaluate batched
        X_pred_cpu = self._batch_predict(val_in)
        X_true = val_out

        better = self.stopping_criterion.update(X_true, X_pred_cpu)

        if better:
            logger.info("Model improved. Storing better model.")
            self._save_best()

    def _train_epoch(self, X: csr_matrix) -> None:
        """Perform a single training epoch
        TODO: which type should X be?
        :param X: user item interaction matrix.
        :type X: csr_matrix
        """
        raise NotImplementedError()

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
            # TODO: It might be possible to put this inner part of the loop
            # into a function,
            # making it possible to overwrite just that part in subclasses.
            in_ = X[batch]
            in_tensor = naive_sparse2tensor(in_).to(self.device)

            out_tensor, _, _ = self.model_(in_tensor)
            batch_results = out_tensor.detach().cpu().numpy()

            results[batch] = batch_results

        logger.info(f"shape of response ({results.shape})")

        return results.tocsr()

    def _transform_training_input(self, X, validation_data):
        """Transform the input matrices of the training function

        :param X: [description]
        :type X: [type]
        :param validation_data: [description]
        :type validation_data: [type]
        :return: [description]
        :rtype: [type]
        """
        return to_csr_matrix((X, validation_data), binary=True)

    def _transform_predict_input(self, X):
        return to_csr_matrix(X, binary=True)

    @property
    def filename(self):
        """Name of the file at which save(self) will write the current best model."""
        return f"{self.name}_loss_{self.stopping_criterion.best_value}.trch"

    def load(self, filename):
        """Load torch model from file.

        :param filename: File to load the model from
        :type filename: str
        """
        with open(filename, "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        """Save the current model to disk.

        filename of the file to save model in is defined by the ``filename`` property.
        """
        with open(self.filename, "wb") as f:
            torch.save(self.model_, f)

    def fit(
        self, X: Matrix, validation_data: Tuple[Matrix, Matrix]
    ) -> "TorchMLAlgorithm":
        """Fit the parameters of the model.

        Interaction Matrix X will be used for training,
        the validation data tuple will be used to compute the evaluate scores.

        :return: **self**, fitted algorithm
        :rtype: TorchMLAlgorithm
        """
        start = time.time()
        # Preconditions:
        # The target for prediction is the validation data.
        assert X.shape == validation_data[0].shape
        assert X.shape == validation_data[1].shape

        # Transform training and validation data to the expected types
        X, validation_data = self._transform_training_input(X, validation_data)

        # Construct variable which will maintain the best model through training
        self.best_model = tempfile.NamedTemporaryFile()

        # Initialize the underlying torch model
        self._init_model(X)

        # TODO Investigate Multi-GPU
        # multi_gpu = False
        # if torch.cuda.device_count() > 1:
        #     self.model_ = torch.nn.DataParallel(self.model_)
        #     multi_gpu = True

        # For each epoch perform a training and evaluation step.
        #  Training uses X
        #  Evaluation uses val_in and val_out
        val_in, val_out = validation_data
        try:
            for epoch in range(self.max_epochs):
                self.model_.train()
                self._train_epoch(X)
                # Make sure no grads are computed while evaluating
                self.model_.eval()
                with torch.no_grad():
                    self._evaluate(val_in, val_out)
        except EarlyStoppingException:
            pass

        self._load_best()

        if self.save_best_to_file:
            self.save()

        # Break this down again, was a local only variable
        self.best_model.close()

        self._check_fit_complete()
        end = time.time()
        logger.info(f"fitting {self.name} complete - Took {start - end :.3}s")

        return self

    def predict(self, X: Matrix) -> csr_matrix:
        """Predict scores for nonzero users in the interaction matrix.

        Predicts items in the interaction matrix in batch.
        Prediction happens in batches, batch sizes in numbers of users
        defined by the batch_size specified during model init.

        Warnings are used if the model is unable to make any recommendations
        for some users.

        :param X: Matrix with interactions to predict from.
        :type X: Matrix
        :return: scores per user item pair
        :rtype: csr_matrix
        """
        check_is_fitted(self)

        X = self._transform_predict_input(X)

        self.model_.eval()
        X_pred = self._batch_predict(X)

        self._check_prediction(X_pred, X)

        return X_pred

    def _check_prediction(self, X_pred: csr_matrix, X: csr_matrix) -> None:
        """Checks that the prediction matches expectations.

        Checks implemented
        - Check that all users with history got at least 1 recommendation

        :param X_pred: The matrix with predictions
        :type X_pred: csr_matrix
        :param X: The input matrix for prediction, used as 'history'
        :type X: csr_matrix
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warnings.warn(
                f"{self.name} failed to recommend any items "
                f"for {len(missing)} users"
            )
