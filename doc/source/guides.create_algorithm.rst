.. _guides-algorithms:

Creating your own algorithms
==============================

Recpack's goal is to make it easy to set up an offline
recommendation experiment to evaluate the merits of a recommendation algorithm.
That is why we've made it super easy to create your own algorithm from scratch,
or adapt from one of the many algorithm implementations included. 

We have created a number of base classes that provide the necessary 
plumbing for your algorithm to fit within the recpack evaluation framework
so that you can focus on what really matters: the algorithm itself!

We explain when and how to use these base classes by means of four example algorithms. 

After reading through these examples you should be able to 

- Know which base class to pick for a new algorithm
- Which functions to overwrite to implement the algorithm

.. _guides-algorithms-pop-softmax:

Randomized Softmax Popularity Algorithm
---------------------------------------

Description
^^^^^^^^^^^

The Softmax Popularity Algorithm samples recommended items relative to their item popularity.
It does so by computing the temperature-weighted softmax of the natural logarithm of the 
item's frequency in the training data. 
With a temperature of 1, we end up with the standard softmax function. 
For a temperature of 0 we always choose the best action, whereas high values of 
temperature starts to resemble uniformly random sampling. 
For more information on the softmax function, check out `Wikipedia <https://en.wikipedia.org/wiki/Softmax_function>`_.

We limit our recommendations to the K most popular items, to avoid ever recommending a very unpopular item.

At prediction time, recommendations are sampled according to their probability (as defined by the softmax). 
The algorithm can thus be considered randomized or stochastic: subsequent calls of the ``predict`` method will
not necessarily yield the same results. 

Implementation
^^^^^^^^^^^^^^

We start by selecting a base class as parent. 
We have no need for PyTorch, so :class:`recpack.algorithms.base.TorchMLAlgorithm` 
is easily discarded.
Likewise, this is neither a factorization nor item-similarity algorithm, 
so we use the base class of base classes: :class:`recpack.algorithms.base.Algorithm`.

Even this base class already implements quite a bit:

- ``fit(X)`` provides a wrapper around the ``_fit`` function we need to implement
  and handles type management and some additional checks.
- ``predict(X)`` provides a wrapper around the ``_predict`` function we need to implement.
- ``_transform_predict_input(X)`` and ``_transform_fit_input(X)`` are used by ``fit`` and ``predict`` 
  to convert their input matrices (X) into the required types. By default, this base class 
  transform the data into a csr_matrix, which suits our purpose perfectly as we have no need 
  of timestamps.
- ``_check_fit_complete()`` is called at the end of the ``fit`` method to make sure 
  fitting was successful.
- ``_check_prediction(X_pred, X)`` is called at the end of ``predict``,
  to make sure the output predicted is as expected. If not it will log a warning.

All we really need to do is thus implement ``__init__`` to set the hyper-parameters,
``_fit`` and ``_predict``.

__init__
""""""""

Let's get started, and define our class, add a docstring for reference and
implement the ``__init__`` function.

Our algorithm has two hyper parameters:

- K: The number of items that can be considered for recommendations.
- Tau: The temperature parameter. 

::

    import numpy as np
    from scipy.sparse import csr_matrix

    from recpack.algorithms.base import Algorithm

    class RandomizedSoftmaxPopularity(Algorithm):
        """Recommend items using softmax on the natural logarithm of item counts.
        
        Recommendations are sampled from the probability distribution
        created by taking the softmax of the natural logarithm of item counts. 
        Items are scored such that the distance between the item in first place
        and the item in second place is the same as between all other items.
        
        :param K: Only the K most frequent items are considered for recommendation
        :param tau: Temperature in the softmax computation
        """
        def __init__(self, K, tau):
            self.K = K
            self.tau = tau

_fit
""""

Next we implement the ``_fit`` function. 
Our input is the matrix of interactions considered for training. 
We compute the natural logarithm of the number of times users interacted 
with the item, then take the softmax of the K most popular items. 

::

    def _fit(self, X: csr_matrix):
        # compute pop by taking logarithm of the raw counts
        #.A1 puts it into a 1d array, making all subsequent operations easy
        pop = np.log(np.sum(X, axis=0)).A1
        
        max_pop = np.max(pop)
        
        # Cut to top K
        self.top_k_pop_items_ = np.argsort(pop)[-self.K:]
        top_k_pop = pop[self.top_k_pop_items_]

        # To make softmax numerically stable, we compute exp((pop - max(pop))/self.tau)
        # instead of exp(pop/self.tau):
        # 
        # softmax for item i can then be computed as 
        # e^((pop[i] - max(pop))/tau) / sum([e^(pop[j] - max(pop))/self.tau for j in topK])
        top_k_pop_minus_max = (top_k_pop - max_pop)/self.tau
        
        top_k_exp = np.exp(top_k_pop_minus_max)
        
        top_k_pop_sum = np.sum(top_k_exp)
        
        self.softmax_scores_ = top_k_exp / top_k_pop_sum

After fitting, the model is ready for prediction.

_predict
""""""""

Finally we implement ``_predict``.
Here we sample recommendations for each user with at least one interaction
in the matrix of interactions. 
Sampling probabilities were stored in ``softmax_scores_`` during fitting.

::

    def _predict(self, X:csr_matrix):
        # Randomly sample items, with weights decided by the softmax scores
        users = X.nonzero()[0]

        # The resulting score = (K - ix)/K
        # The first sampled item gets score 1, and the last sampled item score 1/K
        score_list = [
            (u, i, (self.K-ix)/self.K)
            for u in set(users)
            for ix, i in enumerate(
                np.random.choice(
                    self.top_k_pop_items_,
                    size=self.K,
                    replace=False,
                    p=self.softmax_scores_
                )
            )
        ]
        user_idxs, item_idxs, scores = list(zip(*score_list))
        score_matrix = csr_matrix((scores, (user_idxs, item_idxs)), shape=X.shape)

        return score_matrix

This algorithm can now be used in evaluation pipelines 
just like any other algorithm already available in recpack!

.. _guides-algorithms-recency:

Recency
---------

Description
^^^^^^^^^^^^

Next we create an algorithm that recommends the items that
have been interacted with most recently. 
This algorithm can be considered a baseline, as it is not personalized.


Implementation
^^^^^^^^^^^^^^

Again, we start from :class:`recpack.algorithms.base.Algorithm`.
This new algorithm is different from :ref:`guides-algorithms-pop-softmax` in that 
it needs the time of interaction to be able to make recommendations.
Thankfully, the recpack data format :class:`recpack.data.matrix.InteractionMatrix`
has a ``timestamps`` attribute that stores the time of interaction. 

Our algorithm has no hyperparameters, so we have no use for an ``__init__`` method. 

_transform_fit_input
""""""""""""""""""""

To make sure we receive a :class:`recpack.data.matrix.InteractionMatrix` at fitting time, 
we update ``_transform_fit_input``.

::

    import numpy as np
    from scipy.sparse import csr_matrix, lil_matrix

    from recpack.algorithms.base import Algorithm
    from recpack.data.matrix import InteractionMatrix

    class Recency(Algorithm):
        def _transform_fit_input(self, X):
            # X needs to be an InteractionMatrix for us to have access to
            # the time of interaction at fitting time
            assert isinstance(X, InteractionMatrix)
            # X needs to have timestamps available
            assert X.has_timestamps
            # No transformation needed
            return X

_fit
"""""

Now that we have asserted that ``_fit`` receives an object of type :class:`recpack.data.matrix.InteractionMatrix`,
we fit our algorithm by extracting for each item, its most recent time of interaction.
We then scale this to the interval [0, 1] using minmax normalisation. 

::

    def _fit(self, X:InteractionMatrix):
        # X.timestamps gives a pandas MultiIndex object, indexed by user and item,
        # we drop the index, and group by just the item index
        # then we select the maximal timestamp from this groupby
        max_ts_per_item = X.timestamps.reset_index().groupby('iid')['ts'].max()

        # apply min_max normalisation
        recency = np.zeros(X.shape[1])
        recency[max_ts_per_item.index] = max_ts_per_item.values
        
        most_recent = np.max(recency)
        least_recent = np.min(recency)
        
        recency = (recency - least_recent) / (most_recent - least_recent)
        self.recency_ = recency.copy()

At fitting time, the base class' ``fit`` method calls both ``_transform_fit_input`` and 
``_fit``.
The model is then ready for use, with attribute ``self.recency_`` which contains the recommendation
scores per item.

_predict
"""""""""

Prediction is now easy: for each nonzero user in the input matrix
we set the item's score equal to the recency score we computed in ``_fit``.

::

    def _predict(self, X: csr_matrix):
        results = lil_matrix(X.shape)
        
        users = get_users(X)
        
        results[users] = self.recency_
        
        return results.tocsr()

Here we go, another algorithm ready for use in evaluation!

.. _guides-algorithms-svd:

Singular Value Decomposition
------------------------------

Description
^^^^^^^^^^^^

Let's now implement SVD, a well-known matrix factorization algorithm.
Singular Value Decomposition decomposes a matrix of interactions into three matrices which
when multiplied together approximately reconstructs the original matrix , ``X = U x Sigma X V``.
If matrix ``X`` is of shape ``(|users| x |items|)``,
then ``U`` is of shape ``(|users| x num_components)``,
``Sigma`` is a ``(num_components x num_components)`` matrix,
and finally ``V`` is a ``(num_components x |items|)`` matrix.

Implementation
^^^^^^^^^^^^^^^

Rather than implement the SVD computation ourselves, 
we adapt the optimised TruncatedSVD implementation in sklearn
so that it matches the recpack interface.

As the name suggests, it makes sense to use :class:`recpack.algorithms.base.FactorizationAlgorithm`
as base class in this example.
In addition to the methods implemented in :class:`recpack.algorithms.base.Algorithm` 
which we have highlighted in :ref:`guides-algorithms-pop-softmax`, this class provides:

- ``_predict`` generates recommendations by multiplying the user embeddings of nonzero users with all item embeddings.
- ``_check_fit_complete`` performs an additional check on the dimensions of the embeddings 

All that remains for us to implement is ``__init__`` 
to set hyperparameters and ``_fit`` to compute the embeddings.

__init__
"""""""""

For simplicity we use only one hyperparameter: ``num_components``, which defines the dimension of the embedding.
We also add a parameter ``random_state``, also a parameter of ``TruncatedSVD``, to ensure reproducibility.

.. warning:: 
    The random_state parameter should not be considered a hyperparameter, i.e. we 
    should not perform a parameter search to determine its optimal value.

::

    import numpy as np
    from scipy.sparse import csr_matrix, lil_matrix, diags
    from sklearn.decomposition import TruncatedSVD

    from recpack.algorithms.base import FactorizationAlgorithm

    class SVD(FactorizationAlgorithm):
        """Singular Value Decomposition as dimension reduction recommendation algorithm.

        SVD computed using the TruncatedSVD implementation from sklearn.
        U x Sigma x V = X
        U are the user features, and the item features are computed as Sigma x V.

        :param num_components: The size of the latent dimension
        :type num_components: int

        :param random_state: The seed for the random state to allow for comparison
        :type random_state: int
        """

        def __init__(self, num_components=100, random_state=42):
            super().__init__(num_components=num_components)

            self.random_state = random_state

_fit
"""""

In ``_fit`` we initialize an object of type TruncatedSVD.
For simplicity's sake we expose only ``num_components`` in our algorithm.
All other hyperparameter are left at their default values.

SVD decomposes the matrix into three matrices, while the 
:class:`recpack.algorithms.base.FactorizationAlgorithm` class expects only two: 
a user and item embedding.
Therefore we take the item embedding to be the product of ``Sigma`` and ``V``. 
Since ``Sigma`` is a square matrix this does not change the matrix dimension:
``Sigma x V`` is still a ``(num_components x |items|)`` matrix. 

::

    def _fit(self, X: csr_matrix):
        model = TruncatedSVD(
            n_components=self.num_components, n_iter=7, random_state=self.random_state
        )
        # Factorization computes U x Sigma x V
        # U are the user features,
        # Sigma x V are the item features.
        self.user_embedding_ = model.fit_transform(X)

        V = model.components_
        sigma = diags(model.singular_values_)
        self.item_embedding_ = sigma @ V

        return self


This concludes the modification of the TruncatedSVD algorithm for use in recpack!

.. _guides-algorithms-silly-mf:

SillyMF (Gradient Descent Algorithm)
--------------------------------------

Description
^^^^^^^^^^^^

In this example we implement a very silly, iterative matrix factorization algorithm in PyTorch. 
It is by no means sophisticated or even guaranteed to converge, 
but serves well for our illustration purposes.

The model learns the weights of a matrix factorization of the initial matrix X as 
``X = U x V^T``.

Implementation
^^^^^^^^^^^^^^^
Because we are now dealing with an algorithm optimised
by means of gradient descent, it makes sense to use :class:`recpack.algorithms.base.TorchMLAlgorithm`
as base class in this example.
This base class comes with quite a bit more plumbing that the others:

- ``_predict`` generates recommendations by calling ``_batch_predict`` for batches of users (to keep the memory footprint low).
- ``_check_fit_complete`` performs an additional check of the dimensions of the embeddings.
- ``_check_prediction`` makes sure predictions were made for all nonzero users.
- ``fit(X, validation_data)`` performs a number of training epochs, each followed by an evaluation step on the full dataset. 
    Unlike the other base classes, it now takes an additional ``validation_data`` argument to perform this evaluation step.
- ``save`` saves the current PyTorch model to disk.
- ``load`` loads a PyTorch model from file.
- ``filename`` generates a unique filename for the current best model.
- ``_transform_predict_input`` transforms the input matrix to a ``csr_matrix`` by default.
- ``_transform_fit_input`` transforms the input matrices to a ``csr_matrix`` by default.
-  ``_evaluate`` performs one evaluation step, which consists of making predictions .
    for the validation data and subsequently updating the stopping criterion.
-  ``_load_best`` loads the best model encountered during training as the final model used to make predictions. 
-  ``_save_best`` saves the best model encountered during training to a temporary file.

Which leaves ``__init__``, ``_init_model``, ``_train_epoch``, ``my_loss`` and ``_batch_predict``
for you to implement, as well as the actual PyTorch nn.Module that is your PyTorch model.

MFModule
""""""""

First we create a PyTorch model that encodes this factorization. 
The ``forward`` method is also used to make recommendations at prediction time.

::

    from typing import List

    import numpy as np
    from scipy.sparse import csr_matrix, lil_matrix
    import torch
    import torch.optim as optim
    import torch.nn as nn

    from recpack.algorithms.base import TorchMLAlgorithm
    from recpack.algorithms.stopping_criterion import StoppingCriterion


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

__init__
"""""""""

Next, we create the actual :class:`recpack.algorithms.base.TorchMLAlgorithm`.
The ``__init__`` of any TorchMLAlgorithm expects at least the following
default hyperparameters to be defined:

- ``batch_size`` which dictactes how many users make up a training batch.
- ``max_epochs`` which defines the maximum number of training epochs to run.
- ``learning_rate`` which determines how much to change the model with every update.
- ``stopping_criterion`` to define how to evaluate the model's performance, and if and when to stop early.

We define one additional hyperparameter:

- ``num_components`` which is the dimension of our embeddings for both users and items.

For the sake of example we use a fixed random seed. 
The random seed is set to guarantee reproducibility of results. 

As :class:`recpack.algorithms.stopping_criterion.StoppingCriterion` we use Recall@10.
By default, early stopping is disabled.

::

    class SillyMF(TorchMLAlgorithm):
        def __init__(self, batch_size, max_epochs, learning_rate, num_components=100):
            super().__init__(
                batch_size, 
                max_epochs,
                learning_rate,
                "recall",
                seed=42
            )
            self.num_components = num_components


_init_model
""""""""""""

Next we implement ``_init_model``. We cannot initialize ``MFModule`` as part of SillyMF's
``__init__``, because at this stage, we're unaware of the dimensions of the interaction matrix.

We then define the optimizer.
Here we use simple SGD, but any PyTorch optimizer can be used.

::
            
    def _init_model(self, X:csr_matrix):
        num_users, num_items = X.shape
        self.model_ = MFModule(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)
        
        # We'll use a basic SGD optimiser
        self.optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate)
        self.steps = 0
            
_train_epoch
"""""""""""""

Next we implement training.
First, we need to define a loss function to indicate how well our 
current embeddings are able to perform at the task we set.
As mentioned in the description, this task is to reconstruct the matrix ``X``.
Our loss function computes the average of the absolute error between ``U x V`` 
and the original matrix ``X`` per user.

.. note::
    For an overview of commonly used loss functions, 
    check out `PyTorch <https://pytorch.org/docs/stable/nn.html#loss-functions>`_ loss functions.
            
::

    def my_loss(true_sim, predicted_sim):
        """Computes the total absolute error from predicted compared to true, 
        and averages over all users
        """
        return torch.mean(torch.sum(torch.abs(true_sim - predicted_sim), axis=1))

Now we can continue with ``_train_epoch``. 
Every epoch loops through the entire dataset (most often in batches).
For every batch, the loss and resulting gradients are computed and the embeddings are updated. 

::

    def _train_epoch(self, X):
        losses = []
        item_tensor = torch.arange(X.shape[1]).to(self.device)
        for users in get_batches(get_users(X), batch_size=self.batch_size):
            self.optimizer.zero_grad()
            user_tensor = torch.LongTensor(users).to(self.device)
            scores = self.model_.forward(user_tensor, item_tensor)
            expected_scores = naive_sparse2tensor(X[users])
            loss = my_loss(expected_scores, scores)
            
            # Backwards propagation of the loss
            loss.backward()
            losses.append(loss.item())
            # Update the weight according to the gradients.
            # All automated thanks to torch.
            self.optimizer.step()
            self.steps += 1
        return losses


_batch_predict
"""""""""""""""

Now we can move on to the final step of our implementation: prediction.
Predicting items is the same here as in :ref:`guides-algorithms-svd`: 
the user embeddings of nonzero users are multiplied with item embeddings.

As mentioned in the definition of ``MFModule``, ``MFModule.forward`` is used to make predictions.
This method takes a PyTorch ``Tensor`` of userids and a ``Tensor`` of itemids as inputs.
It then computes the matrix multiplication of its embeddings.

::

    def _batch_predict(self, X: csr_matrix, users: List[int] = None) -> np.ndarray:
        """Predict scores for matrix X, given the selected users.

        If there are no selected users, you can assume X is a full matrix,
        and users can be retrieved as the nonzero indices in the X matrix.

        :param X: Matrix of user item interactions
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: dense matrix of scores per user item pair.
        :rtype: np.ndarray
        """
        X_pred = lil_matrix(X.shape)
        if users is None:
            users = get_users(X)

        # Turn the np arrays and lists to torch tensors
        user_tensor = torch.LongTensor(users).to(self.device)
        item_tensor = torch.arange(X.shape[1]).to(self.device)

        X_pred[users] = self.model_(user_tensor, item_tensor).detach().cpu().numpy()
        return X_pred.tocsr()

And that's all for implementing SillyMF!

.. _guides-algorithms-use-pipeline:

Compare your algorithm to the state of the art
----------------------------------------------

Now that you have learned how to create your own algorithm, you obviously want to know how well it performs compared to state of the art recommendation algorithms.
Recpack provides pipeline functionality, which simplifies running experiments as well as making them reproducible.

Because we want you to use your own algorithms with the recpack pipelines, we have made it easy to set up a pipeline with your own algorithm.

The first (and only) step to using a new algorithm is to make sure it is registered in the `recpack.pipelines.ALGORITHM_REGISTRY`.
Registering a new algorithm is done using the `register` function. This function takes two arguments: the name of the algorithm to register and the class.

::

    from recpack.pipelines import ALGORITHM_REGISTRY

    ALGORITHM_REGISTRY.register(SillyMF.__name__, SillyMF)

Once the algorithm is registered, you can use it when constructing a pipeline.
As an example we will compare the SillyMF algorithm to an ItemKNN algorithm, and the EASE algorithm.

::

    from recpack.data.datasets import MovieLens25M
    from recpack.pipelines import PipelineBuilder
    from recpack.splitters.scenarios import StrongGeneralization
    from recpack.pipelines import ALGORITHM_REGISTRY

    ALGORITHM_REGISTRY.register(SillyMF.__name__, SillyMF)

    # Get data to test on
    dataset = MovieLens25M("data/ml25.csv", preprocess_default=False)
    # This will apply default preprocessing
    im = dataset.load_interaction_matrix()

    # Data splitting scenario
    scenario = StrongGeneralization(frac_users_train=0.7, frac_interactions_in=0.8, validation=True)

    # Construct our pipeline object
    pipeline_builder = PipelineBuilder()

    pipeline_builder.set_train_data(scenario.training_data)
    pipeline_builder.set_test_data(scenario.test_data)
    pipeline_builder.set_validation_data(scenario.validation_data)

    # Add the baseline algorithms
    # Grid parameters will be optimised using grid search before final evaluation
    pipeline_builder.add_algorithm('ItemKNN', grid={'K': [100, 200, 400, 800]})
    pipeline_builder.add_algorithm('EASE', grid={'l2': [10, 100, 1000], 'alpha': [0, 0.1, 0.5]})

    # Add our new algorithm
    # Optimising learning rate and num_components
    # setting fixed values for max_epochs and batch_size
    pipeline_builder.add_algorithm(
        'SillyMF',
        grid={
            'learning_rate': [0.1, 0.01, 0.3], 
            'num_components': [100, 200, 400]
        },
        params={
            'max_epochs': 5,
            'batch_size': 1024
        }
    )

    # Add NDCG and Recall to be evaluated at 10, 20, 50 and 100
    pipeline_builder.add_metric('NormalizedDiscountedCumulativeGainK', [10, 20, 50, 100])
    pipeline_builder.add_metric('RecallK', [10, 20, 50, 100])

    # Set the optimisation metric, this metric will be used to select the best values from grid for each algorithm
    pipeline_builder.set_optimisation_metric('RecallK', 20)

    # Construct pipeline
    pipeline = pipeline_builder.build()

    # Run pipeline, will first do optimisation, and then evaluation
    pipeline.run()

    # Get the metric results.
    # This will be a dict with the results of the run.
    # Turning it into a dataframe makes reading easier
    pd.DataFrame.from_dict(pipeline.get_metrics()).T

And there you have it, hopefully the new algorithm is better than the baseline algorithms!
For more information on how to use pipelines, see :class:`recpack.pipelines`.
