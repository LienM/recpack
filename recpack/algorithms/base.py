import logging
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import tempfile
import torch
from typing import Tuple

import warnings

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

# TODO: needed to duplicate functionality between Algorithm and MetricLearningAlgorithm
# Because fit method had different interface
# Probably should make a super class Algorith, which does not have a fit method.
# And then 2 sub classes ? and MetricLearningAlgorithm,
#   which contain their respective fit interfaces
# This also makes it possible to do a is_subtype check
#   to see what needs to be passed as parameter.


class Algorithm(BaseEstimator):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

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
        # TODO: There might be a way to use this
        # to the advantage of handling csr_matrix or not
        # Eg. 1 base class uses to_csr, the other an InteractionMatrix?
        self._fit(X)

        self._check_fit_complete()
        return self

    def predict(self, X: Matrix) -> csr_matrix:
        check_is_fitted(self)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred


class SimilarityMatrixAlgorithm(Algorithm):
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


class TopKSimilarityMatrixAlgorithm(SimilarityMatrixAlgorithm):
    def __init__(self, K):
        super().__init__()
        self.K = K


class MetricLearningAlgorithm:
    """Base classes for algorithms that optimise a metric,
    and require validation data to select their optimal model.
    """

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def _check_fit_complete(self):
        """Helper function to check that fit stored information."""
        # Use the sklear check_is_fitted function,
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        check_is_fitted(self)

    def fit(
        self, X: Matrix, validation_data: Tuple[Matrix, Matrix]
    ) -> "MetricLearningAlgorithm":
        self._fit(X, validation_data)

        self._check_fit_complete()

        return self

    def predict(self, X: Matrix):
        check_is_fitted(self)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred


class TorchMLAlgorithm(MetricLearningAlgorithm):
    """Base class for algorithms using torch to learn a metric.

    :param MetricLearningAlgorithm: [description]
    :type MetricLearningAlgorithm: [type]
    """

    # TODO: batch predict?

    def __init__(
        self,
        batch_size,
        max_epochs,
        learning_rate,
        stopping_criterion: StoppingCriterion,
        seed=None,
        save_best_to_file=False,
    ):
        # TODO * torch.cuda.device_count() if cuda else batch_size -> Multi GPU
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
        # TODO Give better names
        return f"{self.name}_loss_{self.stopping_criterion.best_value}.trch"

    # TODO: Do we want to keep this? It's kind of useful to reuse the model.
    # But it has been removed from the Algorithm base class.
    def load(self, filename):
        with open(filename, "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        """Save the current model to disk"""
        with open(self.filename, "wb") as f:
            torch.save(self.model_, f)

    def fit(
        self, X: Matrix, validation_data: Tuple[Matrix, Matrix]
    ) -> "TorchMLAlgorithm":
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
        return self

    def predict(self, X: Matrix):
        check_is_fitted(self)

        X = self._transform_predict_input(X)

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
