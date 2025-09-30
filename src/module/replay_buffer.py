"""
Experience Replay Buffer for the Sequence Guessing RL Agent

This module implements experience replay buffer for storing and sampling
transitions from the environment. Supports both primitive actions and
option-based experiences.
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional

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


def create_replay_buffer(config: ReplayBufferConfig) -> ReplayBuffer:
    """
    Factory function to create replay buffer

    Args:
        config: Buffer configuration

    Returns:
        Initialized replay buffer
    """
    return ReplayBuffer(config)