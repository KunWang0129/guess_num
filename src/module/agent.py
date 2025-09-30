"""
DQN Agent with Options Framework Support

This module implements a Deep Q-Network agent using PyTorch Lightning
that supports both primitive actions and hierarchical options.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
import copy
from collections import Counter

from .environment import SequenceGuessingEnv
from .model import create_model, ModelConfig
from .replay_buffer import create_replay_buffer, ReplayBufferConfig
from .options import OptionsManager, create_options_manager, BaseOption, OptionResult


@dataclass
class AgentConfig:
    """Configuration for the DQN Agent"""
    # Learning parameters
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    batch_size: int = 32

    # Exploration parameters
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

    # Target network update
    target_update_freq: int = 100  # Update target network every N steps
    soft_update_tau: float = 0.001  # For soft updates (if enabled)
    use_soft_updates: bool = False

    # Training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 100

    # Q-value normalization
    q_value_clip_max: float = 100.0  # Clip Q-values to prevent explosion

    # Options-specific parameters
    use_macro_options: bool = True
    use_options: Optional[bool] = None  # Deprecated alias
    option_epsilon: float = 0.1  # Separate epsilon for option selection
    option_termination_bonus: float = 0.0  # Bonus for successful option completion

    # Network architecture
    model_type: str = "mlp"  # Only "mlp" is currently supported
    hidden_sizes: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.1

    # Replay buffer
    buffer_size: int = 10000
    min_replay_size: int = 1000
    prioritized_replay: bool = False

    # Options configuration - controls which specific options are enabled
    options: Optional[Dict[str, bool]] = None

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 128]
        if self.use_options is not None:
            self.use_macro_options = self.use_options
            self.use_options = None


class DQNAgent(pl.LightningModule):
    """
    Deep Q-Network Agent with Options Framework Support

    This agent can handle both primitive actions and hierarchical options
    in a unified action selection framework.
    """

    def __init__(self, env: SequenceGuessingEnv, config: AgentConfig):
        """
        Initialize DQN Agent

        Args:
            env: Environment instance
            config: Agent configuration
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = env
        self.config = config

        # Environment information
        env_info = env.get_info()
        self.sequence_length = env_info['sequence_length']
        self.num_values = env_info['num_values']

        # State size for raw states (from environment)
        self.raw_state_size = env_info['state_size']

        # State size for one-hot encoded states (for neural network)
        # One-hot: sequence_length * num_values + 3 metadata features
        self.state_size = self.sequence_length * self.num_values + 3

        self.num_primitive_actions = env_info['action_space_size']

        # Initialize options manager
        if config.use_macro_options:
            # Get enabled options from config if available
            enabled_macros = getattr(config, 'options', None)
            self.options_manager = create_options_manager(
                sequence_length=env_info['sequence_length'],
                num_values=env_info['num_values'],
                enabled_macros=enabled_macros,
                include_primitives=True
            )
            self.num_options = len(self.options_manager.options)
        else:
            self.options_manager = None
            self.num_options = 0

        # Total action space: primitive actions + options
        self.total_actions = self.num_primitive_actions + self.num_options

        # Model configuration
        model_config = ModelConfig(
            state_size=self.state_size,
            num_values=self.num_values,
            num_primitive_actions=self.num_primitive_actions,
            num_options=self.num_options,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            dropout_rate=config.dropout_rate
        )

        # Create networks
        self.q_network = create_model(config.model_type, model_config)
        self.target_network = create_model(config.model_type, model_config)

        # Initialize target network
        self._update_target_network(hard_update=True)

        # Replay buffer
        buffer_config = ReplayBufferConfig(
            capacity=config.buffer_size,
            batch_size=config.batch_size,
            min_size=config.min_replay_size
        )
        self.replay_buffer = create_replay_buffer(buffer_config)

        # Training state
        self.epsilon = config.eps_start
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Current episode state
        self.current_state = None
        self.episode_reward = 0.0
        self.episode_steps = 0

        # Metrics tracking
        self.training_losses = []
        self.q_values_history = []

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

    def _state_to_onehot(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convert raw state representation to one-hot encoded state for neural network input.

        Args:
            state: Raw state tensor (batch_size, raw_state_size) or (raw_state_size,)
                   Format: [sequence values (sequence_length), metadata (3)]

        Returns:
            One-hot encoded state tensor
                   Format: [one-hot encoded sequence (sequence_length * num_values), metadata (3)]
        """
        # Handle both single state and batch
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = state.shape[0]

        # Extract sequence and metadata
        sequence = state[:, :self.sequence_length].long()  # (batch_size, sequence_length)
        metadata = state[:, self.sequence_length:]  # (batch_size, 3)

        # One-hot encode the sequence
        # F.one_hot expects LongTensor and returns (batch_size, sequence_length, num_values)
        sequence_onehot = F.one_hot(sequence, num_classes=self.num_values).float()

        # Flatten the one-hot encoding: (batch_size, sequence_length * num_values)
        sequence_onehot_flat = sequence_onehot.reshape(batch_size, -1)

        # Concatenate with metadata
        onehot_state = torch.cat([sequence_onehot_flat, metadata], dim=1)

        if squeeze_output:
            onehot_state = onehot_state.squeeze(0)

        return onehot_state

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network

        Args:
            state: Raw state tensor (will be converted to one-hot internally)

        Returns:
            Q-values for all actions
        """
        onehot_state = self._state_to_onehot(state)
        return self.q_network(onehot_state)

    def _get_primitive_option_name(self, action_index: int) -> Optional[str]:
        """Get primitive option name from action index"""
        if self.options_manager is None:
            return None

        # Primitive options are the first entries in the catalog
        if 0 <= action_index < len(self.options_manager.primitive_options):
            return self.options_manager.primitive_options[action_index].name
        return None

    def select_action(self, state: torch.Tensor, epsilon: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using epsilon-greedy strategy with options support

        Args:
            state: Current environment state
            epsilon: Epsilon value for exploration (uses self.epsilon if None)

        Returns:
            Tuple of (action, metadata)
        """
        if epsilon is None:
            epsilon = self.epsilon

        env_info = self.env.get_info()

        # Check if we have an active option
        if self.options_manager and self.options_manager.active_option:
            # Execute active option
            option_result = self.options_manager.step_active_option(state, env_info)

            if option_result and option_result.action is not None:
                return option_result.action, {
                    "action_type": "option_step",
                    "option_name": self.options_manager.get_active_option_name(),
                    "option_metadata": option_result.metadata,
                    "option_status": option_result.status.value
                }

        # No active option or option terminated - select new action
        if random.random() < epsilon:
            # Exploration: random action
            if self.config.use_macro_options and random.random() < 0.5:
                # Try to select a random available option
                available_options = self.options_manager.get_available_options(state, env_info)
                if available_options:
                    option_name = random.choice(available_options)
                    if self.options_manager.initiate_option(option_name, state, env_info):
                        # Execute first step of the option
                        option_result = self.options_manager.step_active_option(state, env_info)
                        if option_result and option_result.action is not None:
                            return option_result.action, {
                                "action_type": "option_initiation",
                                "option_name": option_name,
                                "exploration": True,
                                "option_metadata": option_result.metadata
                            }

            # Random primitive action - only select from valid actions
            validity_mask = self.env.get_action_validity_mask()
            valid_actions = torch.where(validity_mask)[0]

            if len(valid_actions) == 0:
                # No valid actions available - should not happen in normal gameplay
                # Fall back to any action (environment will handle this)
                action = random.randint(0, self.num_primitive_actions - 1)
            else:
                action = valid_actions[random.randint(0, len(valid_actions) - 1)].item()

            primitive_option_name = self._get_primitive_option_name(action)
            metadata = {"action_type": "primitive", "exploration": True}
            if primitive_option_name:
                metadata["option_name"] = primitive_option_name
            return action, metadata

        else:
            # Exploitation: use Q-network
            if isinstance(state, torch.Tensor):
                state_tensor = state.to(dtype=torch.float32, device=self.device)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Convert raw state to one-hot for neural network
            state_onehot = self._state_to_onehot(state_tensor)

            with torch.no_grad():
                q_values = self.q_network(state_onehot)

            # Get action validity mask and apply to primitive actions
            validity_mask = self.env.get_action_validity_mask()
            validity_mask_tensor = validity_mask.to(device=self.device)

            # Split Q-values into primitive and option parts
            primitive_q = q_values[:, :self.num_primitive_actions].clone()
            option_q = q_values[:, self.num_primitive_actions:] if self.num_options > 0 else None

            # Apply validity mask to primitive actions
            primitive_q[0, ~validity_mask_tensor] = float('-inf')

            # Select best action overall or best among available options
            if self.config.use_macro_options and option_q is not None:
                # Check which options are available
                available_options = self.options_manager.get_available_options(state, env_info)
                if available_options:
                    # Mask unavailable options
                    option_mask = torch.full_like(option_q, float('-inf'))
                    for i, option_name in enumerate(self.options_manager.options.keys()):
                        if option_name in available_options:
                            option_mask[0, i] = 0

                    masked_option_q = option_q + option_mask

                    # Choose between best valid primitive action and best available option
                    best_primitive_q = primitive_q.max()
                    best_option_q = masked_option_q.max()

                    if best_option_q > best_primitive_q:
                        # Select best available option
                        option_idx = masked_option_q.argmax().item()
                        option_name = list(self.options_manager.options.keys())[option_idx]

                        if self.options_manager.initiate_option(option_name, state, env_info):
                            option_result = self.options_manager.step_active_option(state, env_info)
                            if option_result and option_result.action is not None:
                                return option_result.action, {
                                    "action_type": "option_initiation",
                                    "option_name": option_name,
                                    "exploration": False,
                                    "q_value": best_option_q.item(),
                                    "option_metadata": option_result.metadata
                                }

            # Select best valid primitive action
            action = primitive_q.argmax().item()
            primitive_option_name = self._get_primitive_option_name(action)
            metadata = {
                "action_type": "primitive",
                "exploration": False,
                "q_value": primitive_q.max().item()
            }
            if primitive_option_name:
                metadata["option_name"] = primitive_option_name
            return action, metadata

    def _compute_validity_mask_from_state_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute action validity masks for a batch of states.

        Note: Only masks "no-op" actions (same digit at same index).
        Does NOT mask locked correct positions since target is not in state.

        Args:
            states: Batch of states (batch_size, state_size)

        Returns:
            Boolean mask (batch_size, num_primitive_actions) where True = valid
        """
        batch_size = states.shape[0]
        sequence_length = self.env.sequence_length
        num_values = self.env.num_values

        # Extract current_guess from each state (first sequence_length elements)
        current_guesses = states[:, :sequence_length]  # (batch_size, sequence_length)

        # Initialize all actions as valid
        validity_masks = torch.ones(
            batch_size, self.num_primitive_actions,
            dtype=torch.bool, device=states.device
        )

        # For each action, check if it's a no-op
        for action_idx in range(self.num_primitive_actions):
            position = action_idx // num_values
            value = action_idx % num_values

            # Check if this action would be a no-op for each state in batch
            # Invalid if current_guess[position] == value
            is_no_op = (current_guesses[:, position] == value)
            validity_masks[:, action_idx] = ~is_no_op

        return validity_masks

    def train_step(self) -> Optional[float]:
        """
        Perform one training step

        Returns:
            Training loss or None if can't sample yet
        """
        if not self.replay_buffer.can_sample():
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample_tensors(device=self.device)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Convert raw states to one-hot for neural network input
        states_onehot = self._state_to_onehot(states)
        next_states_onehot = self._state_to_onehot(next_states)

        # Current Q-values
        current_q_values = self.q_network(states_onehot).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network with validity masking
        with torch.no_grad():
            next_q_all = self.target_network(next_states_onehot)

            # Apply validity mask to exclude no-op actions
            validity_masks = self._compute_validity_mask_from_state_batch(next_states)

            # Apply mask to primitive actions only
            next_q_primitives = next_q_all[:, :self.num_primitive_actions].clone()
            next_q_primitives[~validity_masks] = float('-inf')

            # If we have options, also consider them
            if self.num_options > 0:
                next_q_options = next_q_all[:, self.num_primitive_actions:]
                next_q_masked = torch.cat([next_q_primitives, next_q_options], dim=1)
            else:
                next_q_masked = next_q_primitives

            next_q_values = next_q_masked.max(1)[0].detach()
            target_q_values = rewards + (self.config.gamma * next_q_values * (~dones))

            # Clip target Q-values to prevent explosion
            target_q_values = torch.clamp(
                target_q_values,
                -self.config.q_value_clip_max,
                self.config.q_value_clip_max
            )

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (clip each gradient value norm)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.config.target_update_freq == 0:
            self._update_target_network()

        # Update epsilon
        self.epsilon = max(
            self.config.eps_end,
            self.config.eps_start * (self.config.eps_decay ** self.steps_done)
        )

        # Track metrics
        self.training_losses.append(loss.item())

        return loss.item()

    def _update_target_network(self, hard_update: bool = None):
        """Update target network weights"""
        if hard_update is None:
            hard_update = not self.config.use_soft_updates

        if hard_update:
            # Hard update: copy all weights
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update: slowly blend weights
            tau = self.config.soft_update_tau
            for target_param, param in zip(self.target_network.parameters(),
                                         self.q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step"""
        # This method is required by PyTorch Lightning but we handle training manually
        # in our custom training loop since we need environment interaction
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        """Configure optimizer for PyTorch Lightning"""
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        return self.optimizer

    def run_episode(self, train: bool = True) -> Dict[str, Any]:
        """
        Run one complete episode

        Args:
            train: Whether to perform training steps during the episode

        Returns:
            Episode statistics
        """
        state = self.env.reset()
        self.current_state = state
        self.episode_reward = 0.0
        self.episode_steps = 0

        done = False
        step_infos = []
        option_selection_counts = Counter()

        while not done and self.episode_steps < self.config.max_steps_per_episode:
            # Select action
            action, action_metadata = self.select_action(state)

            # Take step in environment
            next_state, reward, done, env_info = self.env.step(action)

            # Store experience in replay buffer (only during training)
            if train:
                is_option = action_metadata.get("action_type") in ["option_initiation", "option_step"]
                option_name = action_metadata.get("option_name")

                self.replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    is_option=is_option,
                    option_name=option_name,
                    metadata=action_metadata
                )

                # Train agent
                loss = self.train_step()
            else:
                loss = None

            # Update state
            state = next_state
            self.episode_reward += reward
            self.episode_steps += 1

            # Log step information
            option_name = action_metadata.get("option_name")
            if option_name:
                # Count both primitive option selections and macro option initiations
                option_selection_counts[option_name] += 1

            step_infos.append({
                "step": self.episode_steps,
                "action": action,
                "action_metadata": action_metadata,
                "reward": reward,
                "done": done,
                "loss": loss,
                "epsilon": self.epsilon,
                "q_values_mean": None  # Could compute if needed
            })

        # Episode finished (only update stats during training)
        if train:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_steps)

        # Clear active option if any
        if self.options_manager:
            self.options_manager.terminate_active_option()

        episode_stats = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_steps,
            "episode_done": done,
            "final_state": state,
            "step_infos": step_infos,
            "replay_buffer_size": len(self.replay_buffer) if train else 0,
            "epsilon": self.epsilon,
            "options_stats": self.options_manager.get_option_stats() if self.options_manager else {},
            "option_selection_counts": dict(option_selection_counts)
        }

        return episode_stats

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent performance over multiple episodes

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation statistics
        """
        self.q_network.eval()

        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        option_counter = Counter()

        with torch.no_grad():
            for episode in range(num_episodes):
                episode_stats = self.run_episode(train=False)
                episode_rewards.append(episode_stats["episode_reward"])
                episode_lengths.append(episode_stats["episode_length"])
                option_counter.update(episode_stats.get("option_selection_counts", {}))

                # Check if episode was successful (sequence guessed correctly)
                if episode_stats.get("episode_done", False):
                    # Check if final state matches target (if available in verbose mode)
                    success_rate += 1

        # Restore training state
        self.q_network.train()
        self.epsilon = original_epsilon

        reward_tensor = torch.tensor(episode_rewards, dtype=torch.float32) if episode_rewards else None
        length_tensor = torch.tensor(episode_lengths, dtype=torch.float32) if episode_lengths else None

        mean_reward = reward_tensor.mean().item() if reward_tensor is not None else 0.0
        std_reward = (
            reward_tensor.std(unbiased=False).item()
            if reward_tensor is not None and reward_tensor.numel() > 1
            else 0.0
        )
        mean_length = length_tensor.mean().item() if length_tensor is not None else 0.0
        std_length = (
            length_tensor.std(unbiased=False).item()
            if length_tensor is not None and length_tensor.numel() > 1
            else 0.0
        )
        best_reward = reward_tensor.max().item() if reward_tensor is not None else 0.0
        worst_reward = reward_tensor.min().item() if reward_tensor is not None else 0.0

        return {
            "num_episodes": num_episodes,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "success_rate": success_rate / num_episodes if num_episodes > 0 else 0.0,
            "best_reward": best_reward,
            "worst_reward": worst_reward,
            "option_selection_counts": dict(option_counter)
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else []
        recent_losses = self.training_losses[-100:] if self.training_losses else []

        reward_tensor = torch.tensor(recent_rewards, dtype=torch.float32) if recent_rewards else None
        length_tensor = torch.tensor(recent_lengths, dtype=torch.float32) if recent_lengths else None
        loss_tensor = torch.tensor(recent_losses, dtype=torch.float32) if recent_losses else None

        recent_reward_mean = reward_tensor.mean().item() if reward_tensor is not None else 0.0
        recent_reward_std = (
            reward_tensor.std(unbiased=False).item()
            if reward_tensor is not None and reward_tensor.numel() > 1
            else 0.0
        )
        recent_length_mean = length_tensor.mean().item() if length_tensor is not None else 0.0
        training_loss_mean = loss_tensor.mean().item() if loss_tensor is not None else 0.0

        return {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.steps_done,
            "current_epsilon": self.epsilon,
            "recent_reward_mean": recent_reward_mean,
            "recent_reward_std": recent_reward_std,
            "recent_length_mean": recent_length_mean,
            "training_loss_mean": training_loss_mean,
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_stats": self.replay_buffer.get_stats(),
            "options_stats": self.options_manager.get_option_stats() if self.options_manager else {}
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint"""
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.steps_done = checkpoint.get("steps_done", 0)
        self.epsilon = checkpoint.get("epsilon", self.config.eps_start)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.training_losses = checkpoint.get("training_losses", [])
