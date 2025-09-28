"""
Experience Replay Buffer for the Sequence Guessing RL Agent

This module implements experience replay buffer for storing and sampling
transitions from the environment. Supports both primitive actions and
option-based experiences.
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
from dataclasses import dataclass, field


def _ensure_state_tensor(data: torch.Tensor) -> torch.Tensor:
    """Clone input data into a float32 CPU tensor for safe storage."""
    if isinstance(data, torch.Tensor):
        return data.detach().clone().to(dtype=torch.float32).cpu()
    return torch.tensor(data, dtype=torch.float32)


@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state: torch.Tensor
    action: int  # Can be primitive action or option ID
    reward: float
    next_state: torch.Tensor
    done: bool
    # Option-specific fields
    is_option: bool = False
    option_name: Optional[str] = None
    option_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer"""
    capacity: int = 10000
    batch_size: int = 32
    min_size: int = 1000  # Minimum size before sampling
    prioritized: bool = False  # Use prioritized experience replay
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4  # Importance sampling exponent
    beta_increment: float = 0.001  # Beta increment per sample
    epsilon: float = 1e-6  # Small constant for numerical stability


class ReplayBuffer:
    """
    Basic Experience Replay Buffer

    Stores experiences and provides random sampling for training.
    Supports both primitive actions and options.
    """

    def __init__(self, config: ReplayBufferConfig):
        """
        Initialize replay buffer

        Args:
            config: Buffer configuration
        """
        self.config = config
        self.buffer = deque(maxlen=config.capacity)
        self.position = 0

    def add(self,
            state: torch.Tensor,
            action: int,
            reward: float,
            next_state: torch.Tensor,
            done: bool,
            is_option: bool = False,
            option_name: Optional[str] = None,
            option_step: int = 0,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add experience to buffer

        Args:
            state: Current state
            action: Action taken (primitive or option ID)
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            is_option: Whether action was an option
            option_name: Name of option if is_option=True
            option_step: Step within option execution
            metadata: Additional information
        """
        experience = Experience(
            state=_ensure_state_tensor(state),
            action=action,
            reward=reward,
            next_state=_ensure_state_tensor(next_state),
            done=done,
            is_option=is_option,
            option_name=option_name,
            option_step=option_step,
            metadata=metadata or {}
        )

        self.buffer.append(experience)

    def sample(self, batch_size: Optional[int] = None) -> List[Experience]:
        """
        Sample batch of experiences

        Args:
            batch_size: Size of batch to sample (uses config default if None)

        Returns:
            List of sampled experiences

        Raises:
            ValueError: If buffer doesn't have enough experiences
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.buffer) < self.config.min_size:
            raise ValueError(f"Buffer has {len(self.buffer)} experiences, "
                           f"need at least {self.config.min_size}")

        return random.sample(self.buffer, batch_size)

    def _experiences_to_tensors(self, experiences: List[Experience], device: str) -> Dict[str, torch.Tensor]:
        """Convert a list of experiences into batched tensors."""
        states = torch.stack([exp.state for exp in experiences]).to(device=device, dtype=torch.float32)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(device=device, dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool, device=device)
        is_options = torch.tensor([exp.is_option for exp in experiences], dtype=torch.bool, device=device)
        option_steps = torch.tensor([exp.option_step for exp in experiences], dtype=torch.long, device=device)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "is_options": is_options,
            "option_steps": option_steps
        }

    def sample_tensors(self, batch_size: Optional[int] = None, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Sample batch and convert to tensors

        Args:
            batch_size: Size of batch to sample
            device: Device to put tensors on

        Returns:
            Dictionary with tensor batches
        """
        experiences = self.sample(batch_size)
        return self._experiences_to_tensors(experiences, device)

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough experiences to sample"""
        if batch_size is None:
            batch_size = self.config.batch_size
        return len(self.buffer) >= max(self.config.min_size, batch_size)

    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all experiences from buffer"""
        self.buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {"size": 0, "capacity": self.config.capacity}

        option_count = sum(1 for exp in self.buffer if exp.is_option)
        primitive_count = len(self.buffer) - option_count

        rewards = [exp.reward for exp in self.buffer]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32) if rewards else None
        avg_reward = reward_tensor.mean().item() if reward_tensor is not None else 0.0
        reward_std = (
            reward_tensor.std(unbiased=False).item() if reward_tensor is not None and reward_tensor.numel() > 1 else 0.0
        )

        return {
            "size": len(self.buffer),
            "capacity": self.config.capacity,
            "option_experiences": option_count,
            "primitive_experiences": primitive_count,
            "option_ratio": option_count / len(self.buffer) if self.buffer else 0,
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "can_sample": self.can_sample()
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer

    Implements prioritized experience replay where experiences are sampled
    based on their TD-error priorities rather than uniformly.
    """

    def __init__(self, config: ReplayBufferConfig):
        """Initialize prioritized replay buffer"""
        super().__init__(config)

        if not config.prioritized:
            raise ValueError("Config must have prioritized=True for PrioritizedReplayBuffer")

        # Sum tree for efficient priority sampling
        tree_capacity = 1
        while tree_capacity < config.capacity:
            tree_capacity *= 2

        self.tree_capacity = tree_capacity
        tree_size = 2 * tree_capacity - 1
        self.sum_tree = torch.zeros(tree_size, dtype=torch.float32)
        self.min_tree = torch.full((tree_size,), float('inf'), dtype=torch.float32)
        self.max_priority = 1.0
        self.beta = config.beta

    def _get_priority(self, error: float) -> float:
        """Convert TD-error to priority"""
        return float((abs(error) + self.config.epsilon) ** self.config.alpha)

    def add(self,
            state: torch.Tensor,
            action: int,
            reward: float,
            next_state: torch.Tensor,
            done: bool,
            is_option: bool = False,
            option_name: Optional[str] = None,
            option_step: int = 0,
            metadata: Optional[Dict[str, Any]] = None,
            priority: Optional[float] = None) -> None:
        """
        Add experience with priority

        Args:
            priority: Priority for this experience (uses max_priority if None)
        """
        # Add to buffer
        super().add(state, action, reward, next_state, done,
                   is_option, option_name, option_step, metadata)

        # Set priority
        if priority is None:
            priority = self.max_priority

        tree_idx = len(self.buffer) - 1 + self.tree_capacity - 1
        self._update_tree(tree_idx, priority)

    def _update_tree(self, tree_idx: int, priority: float) -> None:
        """Update sum tree and min tree with new priority"""
        current_value = self.sum_tree[tree_idx].item()
        change = priority - current_value
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority

        # Propagate changes up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] = self.sum_tree[tree_idx].item() + change
            left = 2 * tree_idx + 1
            right = left + 1
            self.min_tree[tree_idx] = torch.minimum(self.min_tree[left], self.min_tree[right])

    def _sample_index(self, priority_sum: float) -> int:
        """Sample index from sum tree"""
        idx = 0
        while idx < self.tree_capacity - 1:
            left = 2 * idx + 1
            left_sum = self.sum_tree[left].item()
            if priority_sum <= left_sum:
                idx = left
            else:
                priority_sum -= left_sum
                idx = 2 * idx + 2

        buffer_idx = idx - self.tree_capacity + 1
        return min(buffer_idx, len(self.buffer) - 1)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[List[Experience], torch.Tensor, torch.Tensor]:
        """
        Sample batch with importance sampling weights

        Returns:
            Tuple of (experiences, indices, weights)
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.buffer) < self.config.min_size:
            raise ValueError(f"Buffer has {len(self.buffer)} experiences, "
                           f"need at least {self.config.min_size}")

        experiences = []
        indices = []
        weights = []

        # Get priority sum
        priority_segment = self.sum_tree[0].item() / batch_size

        # Update beta
        self.beta = min(1.0, self.beta + self.config.beta_increment)

        # Sample experiences
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            priority_sum = random.uniform(a, b)
            idx = self._sample_index(priority_sum)

            experiences.append(self.buffer[idx])
            indices.append(idx)

            # Calculate importance sampling weight
            tree_idx = idx + self.tree_capacity - 1
            probability = self.sum_tree[tree_idx].item() / self.sum_tree[0].item()
            weight = (len(self.buffer) * probability) ** (-self.beta)
            weights.append(weight)

        # Normalize weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if weights_tensor.numel() > 0:
            max_weight = weights_tensor.max().item()
            if max_weight > 0:
                weights_tensor = weights_tensor / max_weight

        indices_tensor = torch.tensor(indices, dtype=torch.long)

        return experiences, indices_tensor, weights_tensor

    def update_priorities(self, indices: torch.Tensor, errors: torch.Tensor) -> None:
        """
        Update priorities for sampled experiences

        Args:
            indices: Indices of experiences to update
            errors: TD-errors for priority calculation
        """
        for idx_tensor, error_tensor in zip(indices, errors):
            idx = int(idx_tensor)
            error = float(error_tensor)
            priority = self._get_priority(error)
            tree_idx = idx + self.tree_capacity - 1
            self._update_tree(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def sample_tensors(self, batch_size: Optional[int] = None, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Sample batch with importance weights and convert to tensors

        Returns:
            Dictionary with tensor batches including 'weights' and 'indices'
        """
        experiences, indices, weights = self.sample(batch_size)
        result = self._experiences_to_tensors(experiences, device)
        result["weights"] = weights.to(device=device, dtype=torch.float32)
        result["indices"] = indices.to(device=device, dtype=torch.long)
        return result


class SegmentedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with separate segments for different experience types.

    This can help balance sampling between primitive actions and options,
    or between different types of options.
    """

    def __init__(self, config: ReplayBufferConfig, segment_ratios: Optional[Dict[str, float]] = None):
        """
        Initialize segmented replay buffer

        Args:
            config: Buffer configuration
            segment_ratios: Ratios for different segments (e.g., {"primitive": 0.7, "options": 0.3})
        """
        super().__init__(config)

        self.segment_ratios = segment_ratios or {"primitive": 0.5, "options": 0.5}
        self.segments = {name: deque(maxlen=int(config.capacity * ratio))
                        for name, ratio in self.segment_ratios.items()}

    def add(self,
            state: torch.Tensor,
            action: int,
            reward: float,
            next_state: torch.Tensor,
            done: bool,
            is_option: bool = False,
            option_name: Optional[str] = None,
            option_step: int = 0,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add experience to appropriate segment"""
        experience = Experience(
            state=_ensure_state_tensor(state),
            action=action,
            reward=reward,
            next_state=_ensure_state_tensor(next_state),
            done=done,
            is_option=is_option,
            option_name=option_name,
            option_step=option_step,
            metadata=metadata or {}
        )

        # Determine segment
        segment_name = "options" if is_option else "primitive"

        # Add to both main buffer and segment
        self.buffer.append(experience)
        if segment_name in self.segments:
            self.segments[segment_name].append(experience)

    def sample_balanced(self, batch_size: Optional[int] = None) -> List[Experience]:
        """
        Sample experiences with balanced representation from segments

        Args:
            batch_size: Size of batch to sample

        Returns:
            List of sampled experiences
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        experiences = []

        # Calculate samples per segment
        for segment_name, ratio in self.segment_ratios.items():
            if segment_name not in self.segments:
                continue

            segment = self.segments[segment_name]
            if not segment:
                continue

            segment_samples = int(batch_size * ratio)
            if segment_samples > 0:
                available_samples = min(segment_samples, len(segment))
                segment_experiences = random.sample(list(segment), available_samples)
                experiences.extend(segment_experiences)

        # Fill remaining slots with random samples if needed
        remaining = batch_size - len(experiences)
        if remaining > 0 and self.buffer:
            additional = random.sample(list(self.buffer), min(remaining, len(self.buffer)))
            experiences.extend(additional)

        return experiences[:batch_size]

    def get_segment_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each segment"""
        stats = {}
        for name, segment in self.segments.items():
            if segment:
                rewards = [exp.reward for exp in segment]
                reward_tensor = torch.tensor(rewards, dtype=torch.float32) if rewards else None
                avg_reward = reward_tensor.mean().item() if reward_tensor is not None else 0.0
                reward_std = (
                    reward_tensor.std(unbiased=False).item()
                    if reward_tensor is not None and reward_tensor.numel() > 1
                    else 0.0
                )
                stats[name] = {
                    "size": len(segment),
                    "capacity": segment.maxlen,
                    "avg_reward": avg_reward,
                    "reward_std": reward_std
                }
            else:
                stats[name] = {"size": 0, "capacity": segment.maxlen}
        return stats


def create_replay_buffer(config: ReplayBufferConfig) -> ReplayBuffer:
    """
    Factory function to create replay buffer

    Args:
        config: Buffer configuration

    Returns:
        Initialized replay buffer
    """
    if config.prioritized:
        return PrioritizedReplayBuffer(config)
    else:
        return ReplayBuffer(config)