# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
import time
from typing import List, Tuple, Optional
import warnings

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import tempfile
import torch


from recpack.algorithms.stopping_criterion import (
    EarlyStoppingException,
    StoppingCriterion,
)
from recpack.algorithms.util import get_batches, get_users, sample_rows
from recpack.matrix import InteractionMatrix, to_csr_matrix, Matrix
from recpack.util import get_top_K_values


logger = logging.getLogger("recpack")


class Algorithm(BaseEstimator):
    """Base class for all recpack algorithm implementations.

    This class provides fit and predict methods that handle generic
    pre and post-conditions.
    Usually a new algorithm will have to
    implement just the :meth:`_fit` and :meth:`_predict` methods.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """Name of the object.

        Name is made by combining the class name with the parameters
        passed at construction time.

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
        """Stub implementation for fitting an algorithm.

        Will be called by the :meth:`fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Stub for predicting scores to users

        Will be called by the :meth:`predict` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix used as input to predict
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        :return: Predictions made for all nonzero users in X
        :rtype: csr_matrix
        """
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Uses the sklear check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

    def _check_prediction(self, X_pred: csr_matrix, X: csr_matrix) -> None:
        """Checks that the predictions matches expectations

        Checks implemented:

        - Check that all users with history got at least 1 recommendation

        For failing checks a warning is printed.

        :param X_pred: Predictions made for all nonzero users in X
        :type X_pred: csr_matrix
        :param X: User-item interaction matrix used as input to predict
        :type X: csr_matrix
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warnings.warn(f"{self.name} failed to recommend any items " f"for {len(missing)} users")

    def _transform_fit_input(self, X: Matrix) -> csr_matrix:
        """Transform the training data to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix to fit the model to
        :type X: Matrix
        :return: Transformed user-item interaction matrix to fit the model
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _transform_predict_input(self, X: Matrix) -> csr_matrix:
        """Transform the input of predict to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix used as input to predict
        :type X: Matrix
        :return: Transformed user-item interaction matrix used as input to predict
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _assert_is_interaction_matrix(self, *matrices: Matrix) -> None:
        """Make sure that the passed matrices are all an InteractionMatrix."""
        for X in matrices:
            if type(X) != InteractionMatrix:
                raise TypeError(f"{self.name} requires Interaction Matrix as input. Got {type(X)}.")

    def _assert_has_timestamps(self, *matrices: InteractionMatrix):
        """Make sure that the matrices all have timestamp information."""
        for X in matrices:
            if not X.has_timestamps:
                raise ValueError(f"{self.name} requires timestamp information in the InteractionMatrix.")

    def fit(self, X: Matrix) -> "Algorithm":
        """Fit the model to the input interaction matrix.

        After fitting the model will be ready to use for prediction.

        This function will handle some generic bookkeeping
        for each of the child classes,

        - The fit function gets timed, and this will get printed
        - Input data is converted to expected type using call to
          :meth:`_transform_predict_input`
        - The model is trained using the :meth:`_fit` method
        - :meth:`_check_fit_complete` is called to check fitting was succesful

        :param X: The interactions to fit the model on.
        :type X: Matrix
        :return: **self**, fitted algorithm
        :rtype: Algorithm
        """
        start = time.time()
        X = self._transform_fit_input(X)
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")
        return self

    def predict(self, X: Matrix) -> csr_matrix:
        """Predicts scores, given the interactions in X

        Recommends items for each nonzero user in the X matrix.

        This function is a wrapper around the :meth:`_predict` method,
        and performs checks on in- and output data to guarantee proper computation.

        - Checks that model is fitted correctly
        - checks the output using :meth:`_check_prediction` function

        :param X: interactions to predict from.
        :type X: Matrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_matrix
        """
        self._check_fit_complete()

        X = self._transform_predict_input(X)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred


class ItemSimilarityMatrixAlgorithm(Algorithm):
    """Base algorithm for algorithms that fit an item to item similarity model

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.
    """

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_

        # If self.similarity_matrix_ is not a csr matrix,
        # scores will also not be a csr matrix
        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar items for each item

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar items for {missing} items.")


class TopKItemSimilarityMatrixAlgorithm(ItemSimilarityMatrixAlgorithm):
    """Base algorithm for algorithms that fit an item to item similarity model with K similar items for every item

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.

    :param K: How many similar items will be kept per item.
    :type K: int
    """

    def __init__(self, K):
        super().__init__()
        self.K = K


class FactorizationAlgorithm(Algorithm):
    """Base class for factorization algorithms

    During fitting two matrices are constructed.

    - ``user_embedding_`` contains the users embedded in a lower dimensional space,
      shape = ``|users| x num_components``

    - ``item_embedding_`` contains the items embedded in the same dimensions
      shape = ``num_components x |items|``

    Prediction happens by multiplying a user's features with the item embedding.

    Usually a child class will have to
    implement just the :meth:`_fit` method,
    to construct the `self.user_embedding_` and `self.item_embedding_` attributes.

    :param num_components: the dimension of the feature matrices. defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_components=100):
        super().__init__()
        self.num_components = num_components

    def _check_fit_complete(self):
        """Helper function to check that fit stored information.

        Checks implemented:

        - Checks if model is fitted, using the sklearn check_is_fitted function,
          https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        - Checks if `num_components` is correct
        """
        check_is_fitted(self)

        # Post conditions
        assert self.user_embedding_.shape[1] == self.num_components
        assert self.item_embedding_.shape[0] == self.num_components

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in the interaction matrix

        For each nonzero users their embedding is multiplied with the item embeddings,
        and scores are the result of this matrix multiplication.

        Before recommending checks the shape of X to the number of
        items and users in the embeddings.

        :param X: binary interaction matrix.
        :type X: csr_matrix
        :return: matrix with scores for each nonzero user.
        :rtype: csr_matrix
        """
        assert X.shape == (self.user_embedding_.shape[0], self.item_embedding_.shape[1])
        # Get the nonzero users, for these we will recommend.
        users = list(set(X.nonzero()[0]))
        # result is a lil matrix, makes editing rows easy
        result = lil_matrix(X.shape)
        # Set rows of the nonzero users to the predicted scores
        result[users] = self.user_embedding_[users] @ self.item_embedding_

        return result.tocsr()


class TorchMLAlgorithm(Algorithm):
    """Base class for PyTorch algorithms optimized by means of gradient descent/ascent

    Uses gradient descent/ascent with respect to a loss function
    to optimize a model over several epochs of training.

    During evaluation the stopping criterion is used to determine
    which model is best and if it would pay to stop training early
    if the model converges before `max_epochs` have concluded.

    After training the best or last model will be loaded, and used for subsequent prediction.

    The batch size is also used for prediction, to reduce load on GPU memory.

    Usually a child class will have to
    implement the :meth:`_batch_predict`, :meth:`_init_model`
    and :meth:`_train_epoch` methods.

    :param batch_size: How many samples to use in each update step.
        Higher batch sizes make each epoch more efficient,
        but increases the amount of epochs needed to converge to the optimum,
        by reducing the amount of updates per epoch.
    :type batch_size: int
    :param max_epochs: The max number of epochs to train.
        If the stopping criterion uses early stopping, less epochs could be used.
    :type max_epochs: int
    :param learning_rate: How much to update the weights at each update.
    :type learning_rate: float
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :meth:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str
    :param stop_early: If True, early stopping is enabled,
        and after ``max_iter_no_change`` iterations where improvement of loss function
        is below ``min_improvement`` the optimisation is stopped,
        even if max_epochs is not reached.
        Defaults to False
    :type stop_early: bool, optional
    :param max_iter_no_change: If early stopping is enabled,
        stop after this amount of iterations without change.
        Defaults to 5
    :type max_iter_no_change: int, optional
    :param min_improvement: If early stopping is enabled, no change is detected,
        if the improvement is below this value.
        Defaults to 0.01
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used.
        A reasonable sample includes at least 1000 users. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        batch_size: int,
        max_epochs: int,
        learning_rate: float,
        stopping_criterion: str,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: Optional[int] = None,
        validation_sample_size: Optional[int] = None,
    ):
        # TODO batch_size * torch.cuda.device_count() if cuda else batch_size
        #   -> Multi GPU
        self.batch_size = batch_size

        self.max_epochs = max_epochs

        self.learning_rate = learning_rate
        self.stopping_criterion = StoppingCriterion.create(
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
        )
        # Set these values here as well, otherwise identifier will not work
        self.stop_early = stop_early
        self.max_iter_no_change = max_iter_no_change
        self.min_improvement = min_improvement

        if seed is None:
            # Cannot use torch.seed() because numpy requires a seed to be 2**32,
            # whereas torch uses 2**64.
            seed = np.random.get_state()[1][0]

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.save_best_to_file = save_best_to_file

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        self.keep_last = keep_last

        self.predict_topK = predict_topK

        self.validation_sample_size = validation_sample_size

    def _init_model(self, X: Matrix) -> None:
        """Initialise the torch model that will learn the weights

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
        """Load the best model from temp file"""
        self.best_model.seek(0)
        self.model_ = torch.load(self.best_model)

    def _evaluate(self, val_in: Matrix, val_out: Matrix) -> None:
        """Perform evaluation step

        Evaluation computes predictions by passing the ``val_in`` matrix to the model,
        the expected output and predictions are passed to the
        :meth:`recpack.algorithms.stopping_criterion.StoppingCriterion.update` function.
        If the new model is better it is stored using :meth:`_save_best`

        :param val_in: Validation Data input
        :type val_in: csr_matrix
        :param val_out: Expected output from validation data
        :type val_out: csr_matrix
        """
        # Evaluate batched
        val_in = self._transform_predict_input(val_in)
        val_out = to_csr_matrix(val_out)

        if self.validation_sample_size:
            val_in, val_out = sample_rows(val_in, val_out, sample_size=self.validation_sample_size)

        X_pred_cpu = self._predict(val_in)
        # StoppingCriterion expects csr_matrix as output

        better = self.stopping_criterion.update(val_out, X_pred_cpu)

        if better and not self.keep_last:
            logger.info("Model improved. Storing better model.")
            self._save_best()

    def _train_epoch(self, X: Matrix) -> None:
        """Perform a single training epoch

        A training epoch updates the internal model using the provided interactions.
        :param X: user item interaction matrix.
        :type X: csr_matrix
        """
        raise NotImplementedError()

    def _batch_predict(self, X: Matrix, users: List[int]) -> csr_matrix:
        """Predict scores for matrix X, given the selected users in this batch

        :param X: Matrix of user item interactions,
            expected to only contain interactions for those users that are in `users`
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: Sparse matrix of scores per user item pair.
        :rtype: csr_matrix
        """
        raise NotImplementedError("Please implement this function")

    def _get_top_k_recommendations(self, X_pred):
        """Keep only the top K recommendations as configured by the predict_topK hyperparameter

        :param X_pred: Recommendation scores as a sparse matrix.
        :type X_pred: csr_matrix
        :return: The selected recommendation scores as a sparse matrix
        :rtype: csr_matrix
        """
        if self.predict_topK:
            return get_top_K_values(X_pred, self.predict_topK)
        else:
            return X_pred

    def _predict(self, X: Matrix) -> csr_matrix:
        """Compute predictions per batch of users,
        to avoid going out of RAM on the GPU

        Will batch the nonzero users into batches of self.batch_size.

        :param X: The input user interaction matrix
        :type X: csr_matrix
        :return: The predicted affinity of users for items.
        :rtype: csr_matrix
        """

        results = lil_matrix(X.shape)
        self.model_.eval()
        with torch.no_grad():
            for users in get_batches(get_users(X), batch_size=self.batch_size):
                if isinstance(X, InteractionMatrix):
                    batch = X.users_in(users)
                else:
                    batch = lil_matrix(X.shape)
                    batch[users] = X[users]
                    batch = batch.tocsr()

                results[users] = self._get_top_k_recommendations(self._batch_predict(batch, users=users)[users])

        logger.debug(f"shape of response ({results.shape})")

        return results.tocsr()

    def _transform_fit_input(
        self, X: Matrix, validation_data: Tuple[Matrix, Matrix]
    ) -> Tuple[csr_matrix, Tuple[csr_matrix, csr_matrix]]:
        """Transform the input matrices of the training function to the expected types

        All matrices get converted to binary csr matrices

        :param X: The interactions matrix
        :type X: Matrix
        :param validation_data: The tuple with validation_in and validation_out data
        :type validation_data: Tuple[Matrix, Matrix]
        :return: The transformed matrices
        :rtype: Tuple[csr_matrix, Tuple[csr_matrix, csr_matrix]]
        """
        return to_csr_matrix((X, validation_data), binary=True)

    def _transform_predict_input(self, X: Matrix) -> csr_matrix:
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

    def fit(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]) -> "TorchMLAlgorithm":
        """Fit the parameters of the model.

        Interaction Matrix X will be used for training,
        the validation data tuple will be used to compute the evaluate scores.

        This function provides the generic framework for training a PyTorch algorithm,
        such that each child class only needs to implement the
        :meth:`_transform_fit_input`, :meth:`_init_model`, :meth:`_train_epoch`
        and :meth:`_evaluate` functions.

        The function will:

        - Transform input data to the expected types
        - Initialize the model using :meth:`_init_model`
        - Iterate for each epoch until max epochs, or when early stopping conditions are met.

            - Training step using :meth:`_train_epoch`
            - Evaluation step using :meth:`_evaluate`

        Once the model has been fit, the best model is stored to disk,
        if specified during init.

        :return: **self**, fitted algorithm
        :rtype: TorchMLAlgorithm
        """
        start = time.time()
        # Preconditions:
        # The target for prediction is the validation data.
        assert X.shape == validation_data[0].shape
        assert X.shape == validation_data[1].shape

        # Transform training and validation data to the expected types
        X, validation_data = self._transform_fit_input(X, validation_data)

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
                start_time = time.time()
                losses = self._train_epoch(X)
                end_time = time.time()
                logger.info(
                    f"Processed epoch {epoch} in {end_time-start_time :.2f} s."
                    f"Batch Training Loss = {np.mean(losses) :.4f}"
                )
                # Make sure no grads are computed while evaluating
                self.model_.eval()
                with torch.no_grad():
                    start_time = time.time()
                    self._evaluate(val_in, val_out)
                    end_time = time.time()
                    logger.info(f"Evaluation at end of {epoch} took {end_time-start_time :.2f} s.")
        except EarlyStoppingException:
            pass

        if not self.keep_last:
            self._load_best()

        if self.save_best_to_file:
            self.save()

        # Break this down again, was a local only variable
        self.best_model.close()

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")

        return self

    def _check_prediction(self, X_pred: csr_matrix, X: Matrix) -> None:
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
            warnings.warn(f"{self.name} failed to recommend any items " f"for {len(missing)} users")
