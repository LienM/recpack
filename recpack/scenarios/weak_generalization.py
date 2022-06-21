from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import FractionInteractionSplitter


class WeakGeneralization(Scenario):
    """Applies a weak generalization split of the data on a per user basis.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    .. note::
        This scenario duplicates interactions across training, validation
        and test datasets:

    For each user their events ``I_u`` are distributed over the datasets as follows:

    - :attr:`full_training_data` (``IT_u``) contains ``frac_data_in * |I_u|`` (rounded up)
      of the user's interactions in the full dataset.
    - :attr:`validation_training_data` contains ``frac_data_in * |IT_u|`` (rounded up)
      of the user's interactions in the full training dataset.
    - :attr:`validation_data_in` contains the same
      events as :attr:`validation_training_data`
      for users with enough interactions to also
      have one in the validation data out dataset.
    - :attr:`validation_data_in` contains the remaining ``(1 - frac_data_in) * |IT_u|``
      (rounded down) of the user's interactions in the full training dataset.
    - :attr:`test_data_in` contains the same events as :attr:`test_training_data`
      for users with enough interactions to also have an event in the test out dataset.
    - :attr:`test_training_data` contains the remaining ``(1 - frac_data_in) * |I_u|``
      of the user's interactions in the full dataset.

    **Example**

    As an example, splitting following data with ``data_in_frac = 0.5``,
    and ``validation = True``::

        item    0   1   2   3   4   5
        Alice   X   X   X
        Bob             X   X   X   X

    would yield full_training_data::

        item    0   1   2   3   4   5
        Alice   X   X
        Bob             X       X

    validation_training_data::

        item    0   1   2   3   4   5
        Alice       X
        Bob             X

    validation_data_in::

        item    0   1   2   3   4   5
        Alice       X
        Bob             X

    validation_data_out::

        item    0   1   2   3   4   5
        Alice   X
        Bob                     X

    test_data_in::

        item    0   1   2   3   4   5
        Alice   X   X
        Bob             X       X

    test_data_out::

        item    0   1   2   3   4   5
        Alice           X
        Bob                 X       X

    :param frac_data_in: Fraction of interactions per user used for
        training. The interactions are randomly chosen.
    :type frac_data_in: float
    :param validation: Assign a portion of the training dataset to validation data
        if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: The seed to use for the random components of the splitter.
        If None, a random seed will be used. Defaults to None
    :type seed: int, optional
    """

    def __init__(
        self,
        frac_data_in: float,
        validation=False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)

        self.frac_data_in = frac_data_in

        self.interaction_split = FractionInteractionSplitter(self.frac_data_in, seed=self.seed)

        if validation:
            self.validation_splitter = FractionInteractionSplitter(
                self.frac_data_in,
                seed=self.seed,
            )

    def _split(self, data: InteractionMatrix):
        """Splits your data so that a user has interactions in all of
            training, validation and test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        self._full_train_X, self._test_data_out = self.interaction_split.split(data)

        if self.validation:
            (
                self._validation_train_X,
                self._validation_data_out,
            ) = self.validation_splitter.split(self._full_train_X)
            self._validation_data_in = self._validation_train_X.copy()

        self._test_data_in = self._full_train_X.copy()
