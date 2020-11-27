import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from recpack.data.matrix import to_datam
from recpack.splitters.scenarios import StrongGeneralizationTimedMostRecent
from recpack.algorithms.rnn.session_rnn import SessionRNN
from recpack.tests.test_algorithms.util import assert_changed, assert_same


def test_session_rnn_training_epoch():
    # TODO
    pass


def test_session_rnn_evaluation_epoch():
    # TODO
    pass


def test_session_rnn_predict():
    # TODO
    pass
