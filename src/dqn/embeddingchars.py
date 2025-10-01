from typing import List

import numpy as np
import torch
from torch import nn


class EmbeddingChars(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, sequence_list: List[int], hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            sequence_list: list of valid 10-digit sequences (as integers)
            hidden_size: size of hidden layers
        """
        super().__init__()
        seq_width = 10 * 10  # 10 digits (0-9) * 10 positions
        emb_size = 8
        self.embedding_layer = nn.Embedding(obs_size, emb_size)
        self.f0 = nn.Sequential(
            nn.Linear(obs_size*emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, seq_width),
        )
        seq_array = np.zeros((seq_width, len(sequence_list)))
        for i, seq in enumerate(sequence_list):
            seq_str = str(seq).zfill(10)  # Ensure 10 digits with leading zeros
            for j, digit in enumerate(seq_str):
                seq_array[j*10 + int(digit), i] = 1
        self.sequences = torch.Tensor(seq_array)

    def forward(self, x):
        emb = self.embedding_layer(x.int())
        y = self.f0(emb.view(x.shape[0], x.shape[1]*self.embedding_layer.embedding_dim))
        z = torch.tensordot(y, self.sequences.to(self.get_device(x)), dims=[(1,), (0,)])
        return nn.Softmax(dim=1)(z)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index
