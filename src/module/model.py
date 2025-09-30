"""
Neural Network Models for the Sequence Guessing RL Agent

This module defines neural network architectures for the Q-learning agent,
including support for hybrid action spaces (primitive actions + options).

State Representation:
    States are one-hot encoded per position: each position in the sequence
    is represented as a one-hot vector of size num_values, followed by
    metadata features (step_count, episode_done, last_reward).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    # Network architecture
    hidden_sizes: list = None  # [128, 128] by default
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = False

    # Input/output dimensions
    state_size: int = 15  # Will be set based on environment (one-hot encoded: seq_len * num_values + 3)
    num_values: int = 10  # Number of possible values per position (e.g., 10 for digits 0-9)
    num_primitive_actions: int = 100  # sequence_length * num_values (10 * 10)
    num_options: int = 0  # Number of available options - should be set based on enabled options

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 128]


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for Q-value estimation.

    This is a basic neural network that maps states to Q-values for all actions
    (both primitive actions and options).

    Expected input: One-hot encoded state representation where each position
    in the sequence is encoded as a one-hot vector, plus metadata features.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Total action space includes primitive actions + options
        self.total_actions = config.num_primitive_actions + config.num_options

        # Build layers
        layers = []
        input_size = config.state_size

        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(self._get_activation())

            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))

            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, self.total_actions))

        self.network = nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function based on config"""
        if self.config.activation.lower() == "relu":
            return nn.ReLU()
        elif self.config.activation.lower() == "tanh":
            return nn.Tanh()
        elif self.config.activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            state: Tensor of shape (batch_size, state_size)
                   One-hot encoded: [seq_pos_0_onehot, ..., seq_pos_n_onehot, metadata]

        Returns:
            Q-values tensor of shape (batch_size, total_actions)
        """
        return self.network(state)

    def get_primitive_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for primitive actions only"""
        q_values = self.forward(state)
        return q_values[:, :self.config.num_primitive_actions]

    def get_option_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for options only"""
        q_values = self.forward(state)
        return q_values[:, self.config.num_primitive_actions:]


def create_model(model_type: str, config: ModelConfig) -> nn.Module:
    """
    Factory function to create neural network models

    Args:
        model_type: Type of model (currently only "mlp" is supported)
        config: Model configuration

    Returns:
        Initialized neural network model
    """
    if model_type.lower() != "mlp":
        raise ValueError(f"Unknown model type: {model_type}. Only 'mlp' is supported.")

    return SimpleMLP(config)


def calculate_state_size(sequence_length: int, num_values: int) -> int:
    """
    Calculate the state size based on environment configuration

    State uses one-hot encoding per position:
    - Each position in the sequence is encoded as a one-hot vector of size num_values
    - Plus metadata: step_count, episode_done, last_reward (3 values)

    Args:
        sequence_length: Length of the sequence to guess
        num_values: Number of possible values per position (e.g., 10 for digits 0-9)

    Returns:
        Size of the state vector (sequence_length * num_values + 3)
    """
    # State includes:
    # - current_guess: one-hot encoded (sequence_length * num_values)
    # - metadata: step_count, episode_done, last_reward (3)
    return sequence_length * num_values + 3


def calculate_action_space_size(sequence_length: int, num_values: int, num_options: int) -> int:
    """
    Calculate total action space size (primitive actions + options)

    Args:
        sequence_length: Length of the sequence to guess
        num_values: Number of possible digit values (e.g., 10 for digits 0-9)
        num_options: Number of available options

    Returns:
        Total size of the action space
    """
    primitive_actions = sequence_length * num_values
    return primitive_actions + num_options
