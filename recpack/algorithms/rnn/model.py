"""
PyTorch definitions of the neural network models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class GRU4RecTorch(nn.Module):
    """
    PyTorch definition of a basic recurrent neural network for session-based
    recommendations.

    :param num_items: Number of items
    :param hidden_size: Number of neurons in the hidden layer(s)
    :param num_layers: Number of hidden layers
    :param embedding_size: Size of the item embeddings, None for no embeddings
    :param dropout: Dropout applied to embeddings and hidden layers
    :param activation: Final layer activation function, one of "identity", "tanh",
        "softmax", "relu", "elu-<X>", "leaky-<X>"
    """

    def __init__(
        self,
        num_items: int,
        hidden_size: int,
        num_layers: int = 1,
        embedding_size: Optional[int] = None,
        dropout: float = 0,
        activation: str = "elu-1",
    ):
        super().__init__()
        self.input_size = num_items
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = num_items
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(num_items, embedding_size) if embedding_size else None
        self.rnn = nn.GRU(
            embedding_size or num_items,
            hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),
        )
        self.h2o = nn.Linear(hidden_size, num_items)
        self.act = self._create_activation(activation)
        self.init_weights()

    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes scores for each item given an action sequence and previous
        hidden state.

        :param input: Action sequences as a Tensor of shape (A, B)
        :param hidden: Previous hidden state, shape (L, B, H)
        :return: The computed scores for all items at each point in the sequence
                 for every sequence in the batch, as well as the next hidden state.
                 As Tensors of shapes (A*B, I) and (L, B, H) respectively.
        """
        input = (
            self.drop(self.emb(input))
            if self.emb
            else F.one_hot(input, self.input_size).float()
        )
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        output = self.act(self.h2o(output))
        output = output.view(-1, self.output_size)
        return output, hidden

    def init_weights(self) -> None:
        """
        Initializes all model parameters uniformly.
        """
        initrange = 0.01
        for param in self.parameters():
            nn.init.uniform_(param, -initrange, initrange)

    def init_hidden(self, batch_size: int) -> Tensor:
        """
        Returns an initial, zero-valued hidden state with shape (L, B, H).
        """
        weights = next(self.rnn.parameters())
        return weights.new_zeros((self.rnn.num_layers, batch_size, self.hidden_size))

    def _create_activation(self, a):
        if a == "identity":
            return nn.Identity()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax()
        elif a == "relu":
            return nn.ReLU()
        elif a.startswith("elu-"):
            return nn.ELU(alpha=float(a.split("-")[1]))
        elif a.startswith("leaky-"):
            return nn.LeakyReLU(negative_slope=float(a.split("-")[1]))
        raise NotImplementedError("Unsupported activation function")
