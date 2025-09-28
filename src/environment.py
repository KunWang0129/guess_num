
"""
Sequence Guessing Environment

This module implements a custom environment for the sequence guessing game
with a Gym-like API. The environment manages the game state and logic,
handling primary actions from the agent.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch


@dataclass
class SequenceGuessingConfig:
    """Configuration class for the Sequence Guessing Environment"""
    sequence_length: int = 4
    max_episode_steps: int = 100
    num_values: int = 9  # Values 1-9
    reward_correct_guess: float = 100.0
    reward_wrong_guess: float = -1.0
    reward_step: float = -0.1
    reward_partial_correct: float = 10.0
    provide_feedback: bool = True
    allow_repeated_guesses: bool = True
    verbose: bool = False


class CompletionReward:
    """Encapsulates completion-focused reward functions."""

    def __init__(self, reward_correct_guess: float) -> None:
        self.reward_correct_guess_value = reward_correct_guess

    def reward_correct_guess(
        self,
        current_guess: torch.Tensor,
        target_sequence: torch.Tensor,
    ) -> float:
        """Return the reward for correctly guessing the sequence."""
        if torch.equal(current_guess, target_sequence):
            return float(self.reward_correct_guess_value)
        return 0.0


class UtilityReward:
    """Encapsulates optional utility reward functions."""

    _SUPPORTED_REWARDS = ("reward_step",)

    def __init__(self, reward_name: str, *, reward_step: float) -> None:
        if reward_name not in self._SUPPORTED_REWARDS:
            raise ValueError(f"Unsupported utility reward '{reward_name}'")
        self.reward_name = reward_name
        self.reward_step_value = reward_step

    def compute(self) -> float:
        """Compute the configured utility reward."""
        if self.reward_name == "reward_step":
            return float(self.reward_step_value)
        return 0.0


class SequenceGuessingEnv:
    """
    Sequence Guessing Environment with Gym-like API

    The agent must guess a hidden sequence of numbers. Each action represents
    placing a specific value (1-9) at a specific position (0 to sequence_length-1).

    Action Space: position * num_values + (value - 1)
    State Space: Current guess attempts, feedback history, step count
    """

    class SequenceProvider(Protocol):
        """Minimal protocol for sequence providers used by the environment."""

        def next_sequence(
            self, expected_length: Optional[int] = None
        ) -> Tuple[List[int], Dict[str, Any]]:
            ...

    def __init__(
        self,
        config: Optional[SequenceGuessingConfig] = None,
        *,
        sequence_provider: Optional["SequenceGuessingEnv.SequenceProvider"] = None,
        use_utility_reward: bool = False,
        utility_reward_name: Optional[str] = None,
    ):
        """Initialize the Sequence Guessing Environment"""
        self.config = config or SequenceGuessingConfig()
        self.sequence_provider = sequence_provider
        self.completion_reward = CompletionReward(self.config.reward_correct_guess)

        if use_utility_reward:
            selected_utility = utility_reward_name or "reward_step"
            self.utility_reward = UtilityReward(
                reward_name=selected_utility,
                reward_step=self.config.reward_step,
            )
            self.utility_reward_name = selected_utility
        else:
            self.utility_reward = None
            self.utility_reward_name = None

        # Environment parameters
        self.sequence_length = self.config.sequence_length
        self.num_values = self.config.num_values
        self.max_episode_steps = self.config.max_episode_steps

        # Action space: position * num_values + (value - 1)
        # Example: position=0, value=5 -> action = 0*9 + (5-1) = 4
        self.action_space_size = self.sequence_length * self.num_values

        # State space dimensions
        # State includes: current_guess + feedback_history + metadata
        self.state_size = (
            self.sequence_length +  # current guess
            self.sequence_length +  # last feedback (correct positions)
            3  # step_count, episode_done, last_reward
        )

        # Initialize environment state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state variables"""
        self.target_sequence: Optional[torch.Tensor] = None
        self.current_guess = torch.zeros(self.sequence_length, dtype=torch.int64)
        self.step_count = 0
        self.episode_done = False
        self.last_feedback = torch.zeros(self.sequence_length, dtype=torch.int64)
        self.last_reward = 0.0
        self.guess_history: List[torch.Tensor] = []
        self.current_sequence_metadata: Dict[str, Any] = {}

    def reset(self) -> torch.Tensor:
        """
        Reset the environment to start a new episode

        Returns:
            Initial state observation
        """
        self._reset_state()

        if self.sequence_provider is not None:
            sequence_values, metadata = self.sequence_provider.next_sequence(
                expected_length=self.sequence_length
            )
            if len(sequence_values) != self.sequence_length:
                raise ValueError(
                    "Sequence provider returned sequence of length "
                    f"{len(sequence_values)} (expected {self.sequence_length})"
                )
            self.target_sequence = torch.tensor(sequence_values, dtype=torch.int64)
            provider_metadata = dict(metadata or {})
            provider_metadata.setdefault("source", "dataset")
            self.current_sequence_metadata = provider_metadata
        else:
            # Generate new random target sequence (values 1 to num_values)
            self.target_sequence = torch.randint(
                low=1,
                high=self.num_values + 1,
                size=(self.sequence_length,),
                dtype=torch.int64
            )
            self.current_sequence_metadata = {"source": "random"}

        if self.config.verbose:
            print(f"New episode started. Target sequence: {self.target_sequence.tolist()}")

        return self._get_state()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment

        Args:
            action: Integer representing position and value to place
                   Formula: position * num_values + (value - 1)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if not (0 <= action < self.action_space_size):
            raise ValueError(f"Action {action} is not valid. Must be in [0, {self.action_space_size-1}]")

        # Decode action into position and value
        position = action // self.num_values
        value = (action % self.num_values) + 1  # Values are 1-indexed

        # Update current guess at the specified position
        self.current_guess[position] = int(value)
        self.step_count += 1

        # Calculate reward and feedback
        reward = self._calculate_reward()
        self.last_reward = reward

        # Check if episode is done
        done = self._is_episode_done()
        self.episode_done = done

        # Create info dictionary
        info = {
            "step_count": self.step_count,
            "action_decoded": {"position": position, "value": value},
            "current_guess": self.current_guess.clone(),
            "target_sequence": self.target_sequence.clone() if self.config.verbose else None,
            "feedback": self.last_feedback.clone(),
            "guess_history": [guess.clone() for guess in self.guess_history],
            "sequence_metadata": dict(self.current_sequence_metadata),
        }

        if self.config.verbose:
            print(f"Step {self.step_count}: Action {action} -> Pos {position}, Val {value}")
            print(f"Current guess: {self.current_guess.tolist()}, Reward: {reward:.2f}")

        return self._get_state(), reward, done, info

    def _calculate_reward(self) -> float:
        """Calculate reward based on current guess and game state"""
        if self.target_sequence is None:
            raise RuntimeError("Target sequence not initialised. Call reset() first.")

        reward = 0.0
        if self.utility_reward is not None:
            reward += self.utility_reward.compute()

        is_correct = torch.equal(self.current_guess, self.target_sequence)
        reward += self.completion_reward.reward_correct_guess(
            self.current_guess,
            self.target_sequence,
        )

        if is_correct:
            if self.config.verbose:
                print("Correct sequence guessed!")
            return reward

        # Update feedback information without shaping additional rewards
        if self.config.provide_feedback:
            correct_positions = self.current_guess == self.target_sequence
            self.last_feedback = correct_positions.to(dtype=torch.int64)

        if not self.config.allow_repeated_guesses and torch.all(self.current_guess > 0):
            self.guess_history.append(self.current_guess.clone())

        return reward

    def _is_episode_done(self) -> bool:
        """Check if the episode should terminate"""
        if self.target_sequence is None:
            return False

        # Episode ends if sequence is correctly guessed
        if torch.equal(self.current_guess, self.target_sequence):
            return True

        # Episode ends if maximum steps reached
        if self.step_count >= self.max_episode_steps:
            return True

        # Episode ends if repeated guess made (when not allowed)
        if not self.config.allow_repeated_guesses and torch.all(self.current_guess > 0):
            for prev_guess in self.guess_history:
                if torch.equal(self.current_guess, prev_guess):
                    return True

        return False

    def _get_state(self) -> torch.Tensor:
        """
        Get current state representation

        State includes:
        - Current guess (sequence_length)
        - Last feedback (sequence_length) - which positions were correct
        - Metadata: step_count, episode_done, last_reward (3)
        """
        current_guess = self.current_guess.to(dtype=torch.float32)
        last_feedback = self.last_feedback.to(dtype=torch.float32)
        metadata = torch.tensor([
            self.step_count / self.max_episode_steps,
            float(self.episode_done),
            self.last_reward
        ], dtype=torch.float32)

        state = torch.cat([current_guess, last_feedback, metadata])
        return state

    def get_state_with_target(self) -> torch.Tensor:
        """Get state representation augmented with the true target sequence."""
        if self.target_sequence is None:
            raise RuntimeError("Target sequence not initialised. Call reset() first.")

        current_guess = self.current_guess.to(dtype=torch.float32)
        target_sequence = self.target_sequence.to(dtype=torch.float32)
        last_feedback = self.last_feedback.to(dtype=torch.float32)
        metadata = torch.tensor([
            self.step_count / self.max_episode_steps,
            float(self.episode_done),
            self.last_reward
        ], dtype=torch.float32)

        return torch.cat([current_guess, target_sequence, last_feedback, metadata])

    def action_to_position_value(self, action: int) -> Tuple[int, int]:
        """Convert action to position and value"""
        if not (0 <= action < self.action_space_size):
            raise ValueError(f"Action {action} is not valid")

        position = action // self.num_values
        value = (action % self.num_values) + 1
        return position, value

    def position_value_to_action(self, position: int, value: int) -> int:
        """Convert position and value to action"""
        if not (0 <= position < self.sequence_length):
            raise ValueError(f"Position {position} is not valid")
        if not (1 <= value <= self.num_values):
            raise ValueError(f"Value {value} is not valid")

        return position * self.num_values + (value - 1)

    def get_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            "sequence_length": self.sequence_length,
            "num_values": self.num_values,
            "action_space_size": self.action_space_size,
            "state_size": self.state_size,
            "step_count": self.step_count,
            "max_episode_steps": self.max_episode_steps,
            "episode_done": self.episode_done,
            "allow_repeated_guesses": self.config.allow_repeated_guesses,
            "provide_feedback": self.config.provide_feedback,
            "config": self.config,
            "sequence_provider_active": self.sequence_provider is not None,
            "use_utility_reward": self.utility_reward is not None,
            "utility_reward_name": self.utility_reward_name,
        }


def create_env_from_config(config_dict: Dict[str, Any]) -> SequenceGuessingEnv:
    """Create environment from configuration dictionary"""
    # Extract utility reward parameters if present
    use_utility_reward = config_dict.pop('use_utility_reward', False)
    utility_reward_name = config_dict.pop('utility_reward_name', None)
    sequence_provider = config_dict.pop('sequence_provider', None)

    config = SequenceGuessingConfig(**config_dict)
    return SequenceGuessingEnv(
        config,
        sequence_provider=sequence_provider,
        use_utility_reward=use_utility_reward,
        utility_reward_name=utility_reward_name
    )
