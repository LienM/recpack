.. _guides-algorithms:

Creating your own algorithms
==============================

In this guide we will explain how to create or adapt your own algorithm, 
so it can be used with the other recpack functionality.
The recpack algorithms module provides several base classes 
which you can inherit from to make integration easy.
For more info on the base classes, check out :ref:`algorithm-base-classes`.

Through four example algorithms we will explain the concepts of the base classes, 
and give pointers how to work on your own algorithms.

After reading through these examples you should be able to 

- Know which base class to pick for a new algorithm
- Which functions to overwrite to implement the algorithm

.. _guides-algorithms-pop-softmax:

Softmax Popularity algorithm
------------------------------

In this section we create an algorithm that samples from the most popular items, 
with the probabilities defined by applying softmax to the raw popularity scores.
For more info on softmax, check out `Wikipedia <https://en.wikipedia.org/wiki/Softmax_function>`_.

The algorithm computes probabilities, and keeps the K most popular items.
For these most popular items the softmax of their scores is used to get a probability 
for sampling each item.
We use softmax with a temperature parameter, to scale between uniform randomness,
and maximal weight on highest scoring items.

During prediction we will sample for each user K items, 
with the probabilities computed during fitting.

The start of creating a new algorithm is selecting which base class to use as a parrent.
None of the more specific subclasses make sense for our simple popularity 
and randomness based algorithm, so we will use :class:`recpack.algorithms.base.Algorithm`.

The base class already implements quite a few functions:

- ``fit(X)`` -> The fit function provides a wrapper around the ``_fit`` function we will implement,
  and handles type management and some additional checks.
- ``predict(X)`` -> Also a wrapper, this time around the ``_predict`` function we will implement.
- ``_transform_predict_input(X)`` and ``_transform_fit_input(X)``, which ``fit`` and ``predict`` 
  use to put the inputs into the right types. 
  For the baseclass these transform the data into csr_matrices, which we want for our algorithm, 
  so we need not touch these.
- ``_check_fit_complete()`` function at the end of the ``fit`` method to make sure 
  fitting was successful
- ``_check_prediction(X_pred, X)`` function that is called at the end of predict,
  to make sure the output predicted is as expected, and log warnings otherwise.

So all that is left for us to do, is implement ``__init__`` to set the hyper-parameters,
``_fit`` and ``_predict``.

Let's get started, and define our class, add a docstring for reference and
implement the ``__init__`` function.::

    import numpy as np
    from scipy.sparse import csr_matrix

    from recpack.algorithms.base import Algorithm

    class RandomPopularity(Algorithm):
        """Recomend items using softmax on popularity scores.
        
        During recommendation the softmax is taken of the popularity score and subsequent items are
        sampled by their softmax probability, scores are assigned by receding rank
        (such that item sampled first gets highest score)
        
        :param K: How much of the popular items to consider
        :param tau: temperature in the softmax computation, 
            if 1 -> always picks the best action, 0 uniform random.
        """
        def __init__(self, K, tau):
            self.K = K
            self.tau = tau

In our algorithm we have two hyper parameters, K and temperature parameter tau, 
and so these are set during initialisation.

Next step is implementing the ``_fit`` function. 
In this function we will receive a matrix with interactions.
Popularity of an item will be computed as the logarithm of the number of times interacted 
with that item.
Once we have popularity, we will compute the sampling probabilities using softmax 
on the K most popular items. ::

    def _fit(self, X: csr_matrix):
        # compute pop by taking logarithm of the raw counts
        #.A1 puts it into a 1d array, making all subsequent operations easy
        pop = np.log(np.sum(X, axis=0)).A1
        
        max_pop = np.max(pop)
        
        # Cut to top K
        self.top_k_pop_items_ = np.argsort(pop)[-self.K:]
        top_k_pop = pop[self.top_k_pop_items_]

        # To make softmax numerically stable, we will compute exp(pop - max(pop))/self.tau
        # instead of exp(pop)
        # 
        # softmax for item i can then be computed as 
        # e^((pop[i] - max(pop))/tau) / sum([e^(pop[j] - max(pop))/self.tau for j in topK])
        top_k_pop_minus_max = (top_k_pop - max_pop)/self.tau
        
        top_k_exp = np.exp(top_k_pop_minus_max)
        
        top_k_pop_sum = np.sum(top_k_exp)
        
        self.softmax_scores_ = top_k_exp / top_k_pop_sum

After fit has been run, the model will be ready for prediction, and ``self.top_k_pop_items_``
and ``self.softmax_scores_`` are fitted.

Final function to implement is the ``_predict`` function.
In this function we will sample recommendations for each user with at least one interaction
in the interaction matrix. 
Sampling probabilities are defined by the computed ``softmax_scores_``.::

    def _predict(self, X:csr_matrix):
        # Randomly sample items, with weights decided by the softmax scores
        users = X.nonzero()[0]

        # The score will be set as K - ix of sampling
        # The first sampled item will get score K, and the last sampled item score 1
        score_list = [
            (u, i, self.K-ix)
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

We have now defined our algorithm, we can use it to predict scores,
and use it in evaluation pipelines just like any other algorithm already available in RecPack.

.. _guides-algorithms-recency:

Recency
---------
In this section we will create an algorithm that recommends the items that
have been interacted with most recently.

As baseclass we will again use the :class:`recpack.algorithms.base.Algorithm` class.
Our new algorithm is special in that it needs timestamps in order to know which when
items were last visited.
As such we need the ``timestamps`` property from the `recpack.data.DataMatrix` class in the input.
To make sure we receive this class, we will update the ``_transform_fit_input`` to make
sure we get a ``DataMatrix`` object.

We don't have any hyperparameters, our algorithm will just give each item a score
proportional to how long ago the item was last interacted with.

So the first thing to do, is to overwrite the ``_transform_fit_input``. 
We will make the function assert the type and precondition of having timestamps on
the input data. No further transformation is needed.::

    import numpy as np
    from scipy.sparse import csr_matrix, lil_matrix

    from recpack.algorithm.base import Algorithm
    from recpack.data.matrix import InteractionMatrix

    class Recency(Algorithm):
        def _transform_fit_input(self, X):
            # X needs to be an interactionMatrix for it to have timestamps
            assert issubclass(X, InteractionMatrix)
            # X needs to have timestamps available
            assert X.has_timestamps
            # No transformation needed
            return X

Now that we know that the X we receive in ``_fit`` will be of the InteractionMatrix type,
we can fit our algorithm by computing per item it's most recent interaction timestamp.
We will then scale this to the interval [0, 1] using minmax normalisation to avoid
unnecessarily high scores. ::

    def _fit(self, X:InteractionMatrix):
        # data.timestamps gives a pandas MultiIndex object, indexed by user and item,
        # we will drop the index, and group by just the item index
        # Then we select the maximal timestamp from this groupby
        max_ts_per_item = data.timestamps.reset_index().groupby('iid')['ts'].max()

        # apply min_max normalisation
        recency = np.zeros(X.shape[1])
        recency[max_ts_per_item.index] = max_ts_per_item.values
        
        most_recent = np.max(recency)
        least_recent = np.min(recency)
        
        recency = (recency - least_recent) / (most_recent - least_recent)
        self.recency_ = recency.copy()

After calling ``fit``, which will call our just defined ``_transform_fit_input`` and 
``_fit`` functions, 
our model is ready for use, with member ``self.recency_`` containing the recommendation
scores per item.

Prediction is easy, for each nonzero user in the input matrix
we will set each items score equal to the recency score we compouted in the ``_fit`` method.
There is no personalisation, each user will get the same scores.::

    def _predict(self, X: csr_matrix):
        results = lil_matrix(X.shape)
        
        users = get_users(X)
        
        results[users] = self.recency_
        
        return results.tocsr()

And there we go, another algorithm ready for use in evaluation.

.. _guides-algorithms-svd:

Singular Value Decomposition
------------------------------

Let's implement SVD, a well known matrix factorization algorithm.
Singular value decomposition decomposes a matrix of interactions into three matrices which
when multiplied together will approximately reconstruct the original matrix , ``X = U x Sigma X V``.
If matrix ``X`` is of shape ``(|users| x |items|)``,
then ``U`` will be of shape ``(|users| x num_components)``,
``Sigma`` will be a ``(num_components x num_components)`` matrix,
and finally ``V`` will be a ``(num_components x |items|)`` matrix.

Rather than implement the SVD computation ourselves, 
we will rely on the optimised TruncatedSVD implementation in sklearn.

As base class for this algorithm it makes sense to use the 
:class:`recpack.algorithms.base.FactorizationAlgorithm` as the name suggests.
This class provides standard functionality for matrix factorization algorithms.
In addition to the standard functions from :class:`recpack.algorithms.base.Algorithm` 
which we have highlighted in :ref:`guides-algorithms-pop-softmax`, this class provides:

- ``_predict``, prediction always happens in the same way, 
  by multiplying the user embedding with the item embeddings, 
  so that is already implemented in this function
- ``_check_fit_complete`` is extended from the base class, 
  to also check that the dimensions of the embeddings are as expected after fitting.

All that remains for us to implement is the ``__init__`` function 
setting hyperparameters and the ``_fit`` function to compute the embeddings.

For simplicity we will only use one hyperparameter, the num_components. 
This is a required parameter for the ``__init__`` of FactorizationAlgorithm, 
defining the size of the embeddings.
We will also add the parameter `random_state`, which is a parameter of ``TruncatedSVD``, 
and will allow us to control the randomisation in the algorithm.

.. warning:: 
    The random_state parameter should not be considered a hyperparameter. 
    Do not try to optimise it. 
    It's used to guarantee reproducible results not to find a good seed for recommendation.

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

In ``_fit`` we will call use the TruncatedSVD implementation from sklearn, 
for simplicity we don't expose any of its hyperparameters except ``num_components`` in our algorithm, 
and just pick reasonable defaults.

SVD composes the matrix into three matrices, while the 
:class:`recpack.algorithms.base.FactorizationAlgorithm` class expects us to fit 
a user and item embedding.
We will handle this by computing the item embedding by pre multiplying `Sigma` and `V`. 
Since `Sigma` is a square matrix this won't change the size, 
and ``Sigma x V`` is still a ``(num_components x |items|)`` matrix. ::

    def _fit(self, X: csr_matrix):
        model = TruncatedSVD(
            n_components=self.num_components, n_iter=7, random_state=self.random_state
        )
        # Factorization computes U x Sigma x V
        # U are the user features,
        # Sigma x V are the item features.
        self.user_features_ = model.fit_transform(X)

        V = model.components_
        sigma = diags(model.singular_values_)
        self.item_features_ = sigma @ V

        return self

.. _guides-algorithms-silly-mf:

Gradient Descent Algorithm
----------------------------

As example for how to use gradient descent based algorithms using torch with RecPack, 
we will create a kind of silly iterative matrix factorization algorithm.
It's by no means sophisticated or guaranteed to even converge, 
but will serve well for our illustration purposes.

The model tries to learn the weights of a 2 matrix factorization of the initial matrix X, 
``X = U x V``.
The first step is to create a torch model that encodes this factorization. 
This module will be the base model we will fit. 
The forward function will be used to generate recommendations. ::

    import numpy as np
    from scipy.sparse import csr_matrix, lil_matrix
    import torch
    import torch.optim as optim


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

Next step is to define a loss function. 
This loss function will tell how well our estimate of the embeddings in the MFModule
is able to perform at the task we set for it.
In this simple case we want to recreate the original matrix.
Our loss function will compute the average of the absolute error between ``U x V`` 
and the original matrix ``X`` per user.

.. note::
    For better loss functions check out for example Shenbin, Ilya, et al. 
    "RecVAE: A new variational autoencoder for Top-N recommendations with implicit feedback." 
    Proceedings of the 13th International Conference on Web Search and Data Mining. 2020.

::

    def my_loss(true_sim, predicted_sim):
        """Computes the total absolute error from predicted compared to true, 
        and averages over all users
        """
        return torch.mean(torch.sum(torch.abs(true_sim - predicted_sim), axis=1))

Now that we have the loss function and the Module implementation we can create 
a recommendation algorithm.
Since we are using torch to learn a specified loss function, 
it makes sense to use the :class:`recpack.algorithms.base.TorchMLAlgorithm`.
This class helps streamline the process of learning the model iteratively, 
and provides us with a lot of functionality we won't have to create anymore.

- ``fit(X, validation_data)``, unlike the other algorithms we need an additional 
  argument in the fit method.
  The validation data is needed to pick which of the models was best during iteration, 
  this way can pick the model
  that generalizes the best, and avoid overfitting to the training dataset.
  The fit method handles iterating through each of the epochs of training, 
  and potential early stopping.
- ``_transform_fit_input``, this function will overwrite the base one, 
  to also transform the validation data into the required format.
- ``predict(X)``, will call the ``_transform_predict_input`` function and then call 
  the ``_batch_predict`` function.
  The latter is a wrapper around the ``_predict`` method we will implement, 
  to make sure recommendations happen in batches,
  to avoid exceeding RAM usage of a GPU when used.

Remains for us to implement:

- ``_predict``, predicting scores
- ``_train_epoch`` how to perform a training step
- ``_init_model`` initialising our MFModule to start fitting it.

Let's start with ``__init__`` and ``_init_model``, 
we will use the hyperparameters expected by the `TorchMLAlgorithm` class 
and how big our learned embeddings should be.

- ``batch_size`` - how many users to use together in a training batch.
- ``max_epochs`` - How many epochs to train for.
- ``learning_rate`` - How fast should our model's weights be updated.
- ``num_components`` - The size of our embeddings for both users and items.

We will choose the recall@10 as our StoppingCriterion, the StoppingCriterion decides which
of the iterations got the best model, the decision of best model will be based
on the validation data received in the fit method.
For more info on StoppingCriterion and options, see 
:class:`recpack.algorithms.stopping_criterion.StoppingCriterion`.

During ``_init_model`` we will initialise our MFModule based on the received matrix, 
and setup our optimizer.
In this case we'll use SGD, but you could use any other of the torch optimizers.
::

    class SillyMF(TorchMLAlgorithm):
        def __init__(self, batch_size, max_epochs, learning_rate, num_components=100):
            super().__init__(
                batch_size, 
                max_epochs,
                learning_rate,
                StoppingCriterion.create('recall', k=10),
                seed=42
            )
            self.num_components = num_components
            
        def _init_model(self, X:csr_matrix):
            num_users, num_items = X.shape
            self.model_ = MFModule(
                num_users, num_items, num_components=self.num_components
            ).to(self.device)
            
            # We'll use a basic SGD optimiser
            self.optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate)
            self.steps = 0

Predicting items is the same as for the SVD algorithm we defined before, 
user embeddings will be multiplied with item embeddings.
However, here we'll use our MFModule to apply this operation. 
Its ``forward`` method takes a tensor of userids and a tensor of itemids.
It will then compute matrix multiplication of its stored embeddings.
Thus in our ``_predict`` method, we should get the users to predict with, 
and all items, and pass them to the forward method. ::

    def _predict(self, X: csr_matrix, users: List[int] = None) -> np.ndarray:
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

        if users is None:
            users = get_users(X)

        # Turn the np arrays and lists to torch tensors
        user_tensor = torch.LongTensor(users).to(self.device)
        item_tensor = torch.arange(X.shape[1]).to(self.device)

        return self.model_(user_tensor, item_tensor).detach().cpu().numpy()

The final method we should implement is the ``_train_epoch``. 
During each epoch we will compute the predictions for batches of users, 
and then compute the loss on these predicitons compared with our training matrix.
Based on the loss we will let the optimizer update the weights of our embeddings.::

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

And that's it for implementing the torch based matrix factorization.