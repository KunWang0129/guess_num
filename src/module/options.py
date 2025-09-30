"""
Options Framework Implementation

This module implements the Options Framework for the sequence guessing game.
Options define temporally extended actions that consist of multiple primary actions.
Each option provides a policy for selecting actions while it's active.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Sequence
from dataclasses import dataclass
import random
from enum import Enum
import math

import torch

PI_DIGITS = "314159265358979323846264338327950288419716939937510"  # enough digits for typical sequence lengths
def _extract_current_guess(state: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Slice and cast the current guess portion from the state vector."""
    return state[:sequence_length].to(dtype=torch.int64)


def _episode_done(state: torch.Tensor) -> bool:
    """Return whether the episode is marked as done in the state metadata."""
    return bool(state[-2].item())


class OptionStatus(Enum):
    """Status of an option execution"""
    ACTIVE = "active"
    TERMINATED = "terminated"
    INVALID = "invalid"


class OptionKind(Enum):
    """Categorise catalog entries"""
    PRIMITIVE = "primitive"
    MACRO = "macro"


@dataclass(frozen=True)
class PrimitiveAction:
    """Canonical representation of a primitive `(position, value)` guess"""
    position: int
    value: int
    num_values: int

    def __post_init__(self) -> None:
        if not 0 <= self.position:
            raise ValueError("PrimitiveAction position must be non-negative")
        if not 0 <= self.value < self.num_values:
            raise ValueError(
                f"PrimitiveAction value {self.value} out of bounds for num_values={self.num_values}"
            )

    def to_index(self) -> int:
        """Convert primitive action to flat index used by the environment"""
        return self.position * self.num_values + self.value


@dataclass
class OptionResult:
    """Result of an option policy execution"""
    action: Optional[int]  # Primary action index (None if option cannot proceed)
    should_terminate: bool  # Whether option should terminate after this action
    status: OptionStatus  # Current status of the option
    metadata: Dict[str, Any]  # Additional information about the option execution
    primitive_action: Optional[PrimitiveAction] = None  # Structured primitive metadata


@dataclass
class OptionCatalogEntry:
    """Flattened catalog entry combining primitives and macro options"""
    index: int
    name: str
    kind: OptionKind
    option: "BaseOption"
    description: str
    primitive_action: Optional[PrimitiveAction] = None


class BaseOption(ABC):
    """
    Abstract base class for all options in the sequence guessing game.

    An option defines a temporally extended action policy that can guide
    the agent's behavior for multiple steps. Each option must implement:
    - policy: Determines the next primary action to take
    - should_terminate: Checks if the option should end in the current state
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the base option

        Args:
            name: Unique name for this option
            description: Human-readable description of what this option does
        """
        self.name = name
        self.description = description
        self.is_active = False
        self.steps_taken = 0
        self.max_steps = None  # Can be set by subclasses to limit option duration

    @abstractmethod
    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """
        Determine the next primary action according to this option's policy

        Args:
            state: Current environment state
            env_info: Environment information including sequence_length, num_values, etc.

        Returns:
            OptionResult containing the action to take and option status
        """
        pass

    @abstractmethod
    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """
        Check if this option should terminate in the current state

        Args:
            state: Current environment state
            env_info: Environment information

        Returns:
            True if option should terminate, False otherwise
        """
        pass

    def initiate(self) -> None:
        """Initialize the option for execution"""
        self.is_active = True
        self.steps_taken = 0

    def terminate(self) -> None:
        """Terminate the option"""
        self.is_active = False

    def step(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """
        Execute one step of this option

        Args:
            state: Current environment state
            env_info: Environment information

        Returns:
            OptionResult with action and termination info
        """
        if not self.is_active:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.INVALID,
                metadata={"error": "Option not active"}
            )

        # Check if option should terminate
        if self.should_terminate(state, env_info):
            self.terminate()
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "Option termination condition met"}
            )

        # Check max steps limit
        if self.max_steps is not None and self.steps_taken >= self.max_steps:
            self.terminate()
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "Max steps reached"}
            )

        # Get action from policy
        result = self.policy(state, env_info)
        self.steps_taken += 1

        if result.should_terminate:
            self.terminate()

        return result


class PrimitiveOption(BaseOption):
    """Single-step option covering one primitive `(position, value)` guess"""

    def __init__(self, position: int, value: int, num_values: int):
        description = (
            f"Primitive guess: position={position}, value={value}"
        )
        super().__init__(
            name=f"primitive_{position}_{value}",
            description=description
        )
        self.position = position
        self.value = value
        self.num_values = num_values
        self.primitive_action = PrimitiveAction(position, value, num_values)
        self.max_steps = 1


    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        # Primitive options terminate immediately after one step
        return self.steps_taken >= 1

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        metadata = {
            "position": self.position,
            "value": self.value,
            "option_type": OptionKind.PRIMITIVE.value
        }
        return OptionResult(
            action=self.primitive_action.to_index(),
            primitive_action=self.primitive_action,
            should_terminate=True,
            status=OptionStatus.TERMINATED,
            metadata=metadata
        )


class GuessFirstHalfOption(BaseOption):
    """
    Option that only guesses numbers in the first half of the sequence positions.
    For a sequence of length N, this option only considers positions [0, N//2-1].
    """

    def __init__(self):
        super().__init__(
            name="guess_first_half",
            description="Guess up to two distinct positions in the first half of the sequence"
        )
        self.max_steps = 2
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._used_positions = []

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """Terminate when no unused first-half positions remain or episode is done."""
        sequence_length = env_info.get('sequence_length', 4)
        first_half_end = sequence_length // 2
        episode_done = _episode_done(state)

        available_positions = [
            pos for pos in range(first_half_end)
            if pos not in self._used_positions
        ]

        if episode_done:
            return True

        if self.steps_taken >= self.max_steps:
            return True

        return len(available_positions) == 0

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """Choose a random valid position-value pair in the first half"""
        sequence_length = env_info.get('sequence_length', 10)
        num_values = env_info.get('num_values', 10)
        first_half_end = sequence_length // 2

        available_positions = [
            pos for pos in range(first_half_end)
            if pos not in self._used_positions
        ]

        if not available_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No eligible positions in first half"}
            )

        # Choose random position and value
        position = random.choice(available_positions)
        if num_values <= 0:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.INVALID,
                metadata={"reason": "Environment reports no available digit values"}
            )

        value = random.randrange(num_values)

        primitive_action = PrimitiveAction(position, value, num_values)

        self._used_positions.append(position)
        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "available_positions": available_positions,
                "used_positions": list(self._used_positions)
            }
        )


class ComplementaryGuessOption(BaseOption):
    """
    Option that fills complementary positions to make pairs sum to 9.
    For existing guesses at position i in [0, N//2-1], fills position N-1-i
    with a number such that current_guess[N-1-i] + current_guess[i] = 9.
    """

    def __init__(self):
        super().__init__(
            name="palindrome_complement",
            description="Fill mirrored positions so paired digits sum to 9"
        )
        self.max_steps = 2
        self._pairs_completed = 0
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._pairs_completed = 0
        self._used_positions = []

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """Terminate when no valid complements remain or episode is done."""
        sequence_length = env_info.get('sequence_length', 4)
        current_guess = _extract_current_guess(state, sequence_length)
        first_half_end = sequence_length // 2
        episode_done = _episode_done(state)
        num_values = env_info.get('num_values', 9)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        for i in range(first_half_end):
            complement_pos = sequence_length - 1 - i
            if complement_pos in self._used_positions:
                continue
            current_value = current_guess[i].item()
            needed_value = 9 - current_value
            if 0 <= needed_value < num_values:
                return False

        return True

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """Find a complementary position-value pair and make the guess"""
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)
        first_half_end = sequence_length // 2
        # Find valid complementary pairs
        valid_pairs = []
        for i in range(first_half_end):
            complement_pos = sequence_length - 1 - i
            if complement_pos in self._used_positions:
                continue
            current_value = current_guess[i].item()
            needed_value = 9 - current_value
            if 0 <= needed_value < num_values:
                valid_pairs.append((complement_pos, needed_value))

        if not valid_pairs:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No valid complementary pairs available"}
            )

        # Choose a random valid pair
        position, value = random.choice(valid_pairs)

        primitive_action = PrimitiveAction(position, value, num_values)
        if position not in self._used_positions:
            self._used_positions.append(position)

        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE
        if should_end:
            self._pairs_completed += 1
        else:
            self._pairs_completed = self.steps_taken + 1

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "complement_of": sequence_length - 1 - position,
                "original_value": int(current_guess[sequence_length - 1 - position].item()),
                "valid_pairs": valid_pairs,
                "used_positions": list(self._used_positions),
                "pairs_completed": self._pairs_completed
            }
        )


class SameNumberBlockOption(BaseOption):
    """Guess a digit and copy it to an adjacent index to create a block."""

    def __init__(self):
        super().__init__(
            name="same_number_block",
            description="Guess a value and repeat it in an adjacent position"
        )
        self.max_steps = 2
        self._anchor_position: Optional[int] = None
        self._anchor_value: Optional[int] = None
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._anchor_position = None
        self._anchor_value = None
        self._used_positions = []

    def _find_anchor_candidates(self, sequence_length: int) -> List[int]:
        candidates: List[int] = []
        for pos in range(sequence_length):
            if pos in self._used_positions:
                continue
            neighbors = self._neighbor_positions(pos, sequence_length)
            if any(neighbor not in self._used_positions for neighbor in neighbors):
                candidates.append(pos)
        return candidates

    def _neighbor_positions(self, position: int, sequence_length: int) -> List[int]:
        neighbors: List[int] = []
        if position > 0:
            neighbors.append(position - 1)
        if position < sequence_length - 1:
            neighbors.append(position + 1)
        return neighbors

    def _available_neighbors(self, position: int, sequence_length: int) -> List[int]:
        neighbors = self._neighbor_positions(position, sequence_length)
        return [neighbor for neighbor in neighbors if neighbor not in self._used_positions]

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        if self._anchor_position is None:
            return len(self._find_anchor_candidates(sequence_length)) == 0

        return len(self._available_neighbors(self._anchor_position, sequence_length)) == 0

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        if self._anchor_position is None:
            candidates = self._find_anchor_candidates(sequence_length)
            if not candidates:
                return OptionResult(
                    action=None,
                    should_terminate=True,
                    status=OptionStatus.TERMINATED,
                    metadata={"reason": "No positions with an available neighbor"}
                )

            position = random.choice(candidates)
            if num_values <= 0:
                return OptionResult(
                    action=None,
                    should_terminate=True,
                    status=OptionStatus.INVALID,
                    metadata={"reason": "Environment reports no available digit values"}
                )

            value = random.randrange(num_values)
            self._anchor_position = position
            self._anchor_value = value
            self._used_positions.append(position)
            primitive_action = PrimitiveAction(position, value, num_values)

            return OptionResult(
                action=primitive_action.to_index(),
                primitive_action=primitive_action,
                should_terminate=False,
                status=OptionStatus.ACTIVE,
                metadata={
                    "stage": "anchor",
                    "position": position,
                    "value": value,
                    "candidates": candidates
                }
            )

        neighbors = self._available_neighbors(self._anchor_position, sequence_length)
        if not neighbors or self._anchor_value is None:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No available neighbor to complete block"}
            )

        anchor_pos = self._anchor_position
        anchor_value = self._anchor_value
        neighbor_position = random.choice(neighbors)
        primitive_action = PrimitiveAction(neighbor_position, anchor_value, num_values)
        self._used_positions.append(neighbor_position)
        self._anchor_position = None
        self._anchor_value = None

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=True,
            status=OptionStatus.TERMINATED,
            metadata={
                "stage": "expand",
                "anchor_position": anchor_pos,
                "neighbor_position": neighbor_position,
                "value": anchor_value,
                "available_neighbors": neighbors
            }
        )


class GaussianGuessOption(BaseOption):
    """Sample digits from a Gaussian distribution centred at five."""

    def __init__(self, mean: float = 5.0, variance: float = 2.0):
        std = math.sqrt(variance)
        super().__init__(
            name="gaussian_fill",
            description="Sample guesses from N(5, 2) for up to two actions"
        )
        self.mean = mean
        self.std = std
        self.max_steps = 2
        self._chosen_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._chosen_positions = []

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        sequence_length = env_info.get('sequence_length', 4)
        return len(self._chosen_positions) >= sequence_length

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)

        available_positions = [
            pos for pos in range(sequence_length)
            if pos not in self._chosen_positions
        ]

        if not available_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No positions remaining for Gaussian sampling"}
            )

        if num_values <= 0:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.INVALID,
                metadata={"reason": "Environment reports no available digit values"}
            )

        raw_value = random.gauss(self.mean, self.std)
        value = int(round(raw_value))
        value = max(0, min(num_values - 1, value))

        position = random.choice(available_positions)
        self._chosen_positions.append(position)

        primitive_action = PrimitiveAction(position, value, num_values)
        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "raw_sampled_value": raw_value,
                "gaussian_params": {"mean": self.mean, "variance": self.std ** 2},
                "available_positions": available_positions
            }
        )


class MonotonicSequenceFillOption(BaseOption):
    """Fill positions with values strictly larger than their predecessors."""

    def __init__(self):
        super().__init__(
            name="monotonic_sequence_fill",
            description="Fill indices with values larger than their immediate predecessors"
        )
        self.max_steps = 2
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._used_positions = []

    def _candidate_positions(self, current_guess: torch.Tensor, num_values: int) -> List[int]:
        sequence_length = len(current_guess)
        candidates: List[int] = []
        for pos in range(1, sequence_length):
            if pos in self._used_positions:
                continue
            prev_value = int(current_guess[pos - 1].item())
            if prev_value >= num_values - 1:
                continue
            candidates.append(pos)
        return candidates

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        return len(self._candidate_positions(current_guess, num_values)) == 0

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)

        candidates = self._candidate_positions(current_guess, num_values)
        if not candidates:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No eligible positions with larger-than-previous values"}
            )

        position = random.choice(candidates)
        prev_value = current_guess[position - 1].item()
        possible_values = list(range(prev_value + 1, num_values))

        if not possible_values:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No values larger than predecessor available"}
            )

        value = random.choice(possible_values)
        primitive_action = PrimitiveAction(position, value, num_values)

        self._used_positions.append(position)
        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "previous_value": prev_value,
                "possible_values": possible_values,
                "used_positions": list(self._used_positions)
            }
        )


class SumTargeterOption(BaseOption):
    """Choose digits that move the total sum closer to sixty."""

    TARGET_SUM = 60

    def __init__(self):
        super().__init__(
            name="sum_targeter",
            description="Select values that steer the sequence sum toward 60"
        )
        self.max_steps = 2
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._used_positions = []

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        current_guess = _extract_current_guess(state, sequence_length)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        return len(self._available_positions(sequence_length)) == 0

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)

        available_positions = self._available_positions(sequence_length)

        if not available_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No unused positions available"}
            )

        if num_values <= 0:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.INVALID,
                metadata={"reason": "Environment reports no available digit values"}
            )

        position = random.choice(available_positions)
        current_sum_tensor = current_guess.sum()
        current_sum = int(current_sum_tensor.item()) if isinstance(current_sum_tensor, torch.Tensor) else int(current_sum_tensor)

        best_value = 0
        best_diff = float("inf")
        for value in range(num_values):
            diff = abs((current_sum + value) - self.TARGET_SUM)
            if diff < best_diff or (diff == best_diff and value < best_value):
                best_value = value
                best_diff = diff

        primitive_action = PrimitiveAction(position, best_value, num_values)
        self._used_positions.append(position)

        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": best_value,
                "current_sum": current_sum,
                "target_sum": self.TARGET_SUM,
                "used_positions": list(self._used_positions)
            }
        )

    def _available_positions(self, sequence_length: int) -> List[int]:
        return [
            pos for pos in range(sequence_length)
            if pos not in self._used_positions
        ]


class HighLowAlternatorOption(BaseOption):
    """Alternate low guesses on even indices and high guesses on odd indices."""

    EVEN_VALUES = (0, 1, 2)
    ODD_VALUES = (7, 8, 9)

    def __init__(self):
        super().__init__(
            name="high_low_alternator",
            description="Even positions choose low digits, odd positions choose high digits"
        )
        self.max_steps = 2
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._used_positions = []

    def _allowed_values(self, position: int, num_values: int) -> List[int]:
        base_values = self.EVEN_VALUES if position % 2 == 0 else self.ODD_VALUES
        values = [v for v in base_values if 0 <= v < num_values]
        if not values:
            if num_values <= 0:
                return []
            fallback = 0 if position % 2 == 0 else num_values - 1
            values = [fallback]
        return values

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        for pos in range(sequence_length):
            if pos in self._used_positions:
                continue
            if self._allowed_values(pos, num_values):
                return False

        return True

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)

        candidates = []
        for pos in range(sequence_length):
            if pos in self._used_positions:
                continue
            allowed = self._allowed_values(pos, num_values)
            if allowed:
                candidates.append((pos, allowed))

        if not candidates:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No positions satisfy alternating high/low constraints"}
            )

        position, allowed_values = random.choice(candidates)
        value = random.choice(allowed_values)
        primitive_action = PrimitiveAction(position, value, num_values)
        self._used_positions.append(position)

        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "allowed_values": allowed_values,
                "used_positions": list(self._used_positions)
            }
        )


class Pattern84Option(BaseOption):
    """Fill two adjacent indices with the repeating pattern '8-4'."""

    PATTERN: Sequence[int] = (8, 4)

    def __init__(self):
        super().__init__(
            name="pattern_84_fill",
            description="Apply the '84' pattern to adjacent indices"
        )
        self.max_steps = 2
        self._start_position: Optional[int] = None
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._start_position = None
        self._used_positions = []

    def _available_starts(self, sequence_length: int) -> List[int]:
        candidates: List[int] = []
        for pos in range(sequence_length - 1):
            if pos in self._used_positions or (pos + 1) in self._used_positions:
                continue
            candidates.append(pos)
        return candidates

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        if max(self.PATTERN) >= num_values:
            return True

        if self._start_position is None:
            return len(self._available_starts(sequence_length)) == 0

        next_index = self._start_position + 1
        if next_index >= sequence_length:
            return True

        return next_index in self._used_positions

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        if max(self.PATTERN) >= num_values:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "Pattern digits exceed available values"}
            )

        if self._start_position is None:
            candidates = self._available_starts(sequence_length)
            if not candidates:
                return OptionResult(
                    action=None,
                    should_terminate=True,
                    status=OptionStatus.TERMINATED,
                    metadata={"reason": "No adjacent positions available for pattern"}
                )

            start = random.choice(candidates)
            self._start_position = start
            value = self.PATTERN[0]
            self._used_positions.append(start)
            primitive_action = PrimitiveAction(start, value, num_values)

            return OptionResult(
                action=primitive_action.to_index(),
                primitive_action=primitive_action,
                should_terminate=False,
                status=OptionStatus.ACTIVE,
                metadata={
                    "stage": "start",
                    "start_position": start,
                    "value": value,
                    "pattern": self.PATTERN,
                    "candidates": candidates
                }
            )

        next_index = self._start_position + 1
        if next_index >= sequence_length or next_index in self._used_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "Second position unavailable for pattern"}
            )

        value = self.PATTERN[1]
        self._used_positions.append(next_index)
        start_position = self._start_position
        primitive_action = PrimitiveAction(next_index, value, num_values)
        self._start_position = None

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=True,
            status=OptionStatus.TERMINATED,
            metadata={
                "stage": "finish",
                "start_position": start_position,
                "end_position": next_index,
                "value": value,
                "pattern": self.PATTERN
            }
        )


class PiDigitsOption(BaseOption):
    """Fill positions using digits of π based on their indices."""

    def __init__(self):
        super().__init__(
            name="pi_digits_fill",
            description="Assign digits from π to the lowest available indices"
        )
        self.max_steps = 2
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._used_positions = []

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        return len(self._available_positions(sequence_length)) == 0

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        available_positions = self._available_positions(sequence_length)

        if not available_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No positions remain for π digits"}
            )

        if num_values <= 0:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.INVALID,
                metadata={"reason": "Environment reports no available digit values"}
            )

        position = min(available_positions)
        digit_char = PI_DIGITS[position % len(PI_DIGITS)]
        raw_value = int(digit_char)
        value = raw_value % num_values

        primitive_action = PrimitiveAction(position, value, num_values)
        self._used_positions.append(position)

        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "digit_source": digit_char,
                "used_positions": list(self._used_positions)
            }
        )

    def _available_positions(self, sequence_length: int) -> List[int]:
        return [
            pos for pos in range(sequence_length)
            if pos not in self._used_positions
        ]


class DifferenceMaintainerOption(BaseOption):
    """Keep new guesses within two of both the current min and max digits."""

    def __init__(self):
        super().__init__(
            name="difference_maintainer",
            description="Ensure new digits stay within two of the current min/max"
        )
        self.max_steps = 2
        self._used_positions: List[int] = []

    def initiate(self) -> None:
        super().initiate()
        self._used_positions = []

    def _valid_values(self, digits: List[int], num_values: int) -> List[int]:
        if not digits:
            return list(range(num_values))

        min_val = min(digits)
        max_val = max(digits)
        return [
            value for value in range(num_values)
            if abs(value - min_val) <= 2 and abs(value - max_val) <= 2
        ]

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)
        episode_done = _episode_done(state)

        if episode_done or self.steps_taken >= self.max_steps:
            return True

        digits = [int(value) for value in current_guess.tolist()]
        valid_values = self._valid_values(digits, num_values)
        if not valid_values:
            return True

        return len(self._available_positions(sequence_length)) == 0

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)

        available_positions = self._available_positions(sequence_length)

        if not available_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No positions available"}
            )

        digits = [int(value) for value in current_guess.tolist()]
        valid_values = self._valid_values(digits, num_values)

        if not valid_values:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "Cannot maintain difference constraints"}
            )

        position = random.choice(available_positions)
        value = random.choice(valid_values)
        primitive_action = PrimitiveAction(position, value, num_values)
        self._used_positions.append(position)

        should_end = (self.steps_taken + 1) >= self.max_steps
        status = OptionStatus.TERMINATED if should_end else OptionStatus.ACTIVE

        min_val = min(digits) if digits else None
        max_val = max(digits) if digits else None

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=should_end,
            status=status,
            metadata={
                "position": position,
                "value": value,
                "current_min": min_val,
                "current_max": max_val,
                "valid_values": valid_values,
                "used_positions": list(self._used_positions)
            }
        )

    def _available_positions(self, sequence_length: int) -> List[int]:
        return [
            pos for pos in range(sequence_length)
            if pos not in self._used_positions
        ]


class OptionsManager:
    """Manage primitive catalog entries and macro options in a unified registry"""

    def __init__(
        self,
        primitive_options: Optional[List[PrimitiveOption]] = None,
        macro_options: Optional[List[BaseOption]] = None
    ):
        self.primitive_options = primitive_options or []
        self.options = {opt.name: opt for opt in (macro_options or [])}
        self.active_option: Optional[BaseOption] = None
        self.active_option_name: Optional[str] = None
        self.option_history: List[str] = []
        self.catalog: List[OptionCatalogEntry] = []
        self.name_to_index: Dict[str, int] = {}
        self._build_catalog()

    # ------------------------------------------------------------------
    # Catalog utilities
    # ------------------------------------------------------------------
    def _build_catalog(self) -> None:
        self.catalog.clear()
        self.name_to_index.clear()

        index = 0
        for primitive in self.primitive_options:
            entry = OptionCatalogEntry(
                index=index,
                name=primitive.name,
                kind=OptionKind.PRIMITIVE,
                option=primitive,
                description=primitive.description,
                primitive_action=primitive.primitive_action
            )
            self.catalog.append(entry)
            self.name_to_index[primitive.name] = index
            index += 1

        for option in self.options.values():
            entry = OptionCatalogEntry(
                index=index,
                name=option.name,
                kind=OptionKind.MACRO,
                option=option,
                description=option.description,
                primitive_action=None
            )
            self.catalog.append(entry)
            self.name_to_index[option.name] = index
            index += 1

    def catalog_size(self) -> int:
        return len(self.catalog)

    def get_catalog(self) -> List[OptionCatalogEntry]:
        return list(self.catalog)

    def get_entry(self, index: int) -> OptionCatalogEntry:
        return self.catalog[index]

    def get_index(self, name: str) -> Optional[int]:
        return self.name_to_index.get(name)

    def is_primitive(self, index: int) -> bool:
        return self.catalog[index].kind == OptionKind.PRIMITIVE

    def get_primitive_action(self, index: int) -> Optional[PrimitiveAction]:
        entry = self.get_entry(index)
        return entry.primitive_action if entry.kind == OptionKind.PRIMITIVE else None

    # ------------------------------------------------------------------
    # Macro option lifecycle
    # ------------------------------------------------------------------
    def get_available_options(self, state: torch.Tensor, env_info: Dict[str, Any]) -> List[str]:
        """Return macro option names that can start under the current state"""
        return [
            name for name, option in self.options.items()
        ]

    def initiate_option(self, option_name: str, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        if option_name not in self.options:
            return False

        option = self.options[option_name]

        if self.active_option:
            self.active_option.terminate()

        option.initiate()
        self.active_option = option
        self.active_option_name = option_name
        self.option_history.append(option_name)
        return True

    def step_active_option(self, state: torch.Tensor, env_info: Dict[str, Any]) -> Optional[OptionResult]:
        if self.active_option is None:
            return None

        result = self.active_option.step(state, env_info)

        if result.should_terminate or result.status == OptionStatus.TERMINATED:
            self.active_option.terminate()
            self.active_option = None
            self.active_option_name = None

        return result

    def terminate_active_option(self) -> None:
        if self.active_option:
            self.active_option.terminate()
            self.active_option = None
            self.active_option_name = None

    def get_active_option_name(self) -> Optional[str]:
        return self.active_option_name

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_option_stats(self) -> Dict[str, Any]:
        from collections import Counter

        return {
            "total_options_used": len(self.option_history),
            "option_usage_counts": dict(Counter(self.option_history)),
            "currently_active": self.get_active_option_name(),
            "available_options": list(self.options.keys()),
            "primitive_catalog_size": len(self.primitive_options),
            "catalog_size": self.catalog_size()
        }


def create_default_macro_options() -> List[BaseOption]:
    """Create the default set of temporally extended macro options"""
    return [
        GuessFirstHalfOption(),
        ComplementaryGuessOption(),
        SameNumberBlockOption(),
        GaussianGuessOption(),
        MonotonicSequenceFillOption(),
        SumTargeterOption(),
        HighLowAlternatorOption(),
        Pattern84Option(),
        PiDigitsOption(),
        DifferenceMaintainerOption()
    ]


def create_default_options() -> List[BaseOption]:
    """Backward-compatible alias for default macro options"""
    return create_default_macro_options()


def create_primitive_option_catalog(sequence_length: int, num_values: int) -> List[PrimitiveOption]:
    """Enumerate the full `(position, value)` primitive catalog"""
    primitive_options: List[PrimitiveOption] = []
    for position in range(sequence_length):
        for value in range(num_values):
            primitive_options.append(PrimitiveOption(position, value, num_values))
    return primitive_options


def create_options_manager(
    sequence_length: Optional[int] = None,
    num_values: Optional[int] = None,
    enabled_macros: Optional[Dict[str, bool]] = None,
    include_primitives: bool = True
) -> OptionsManager:
    """Create an OptionsManager configured for the environment layout"""

    primitive_options: List[PrimitiveOption] = []
    if include_primitives and sequence_length is not None and num_values is not None:
        primitive_options = create_primitive_option_catalog(sequence_length, num_values)

    macro_options = create_default_macro_options()
    if enabled_macros is not None:
        macro_options = [
            option for option in macro_options
            if enabled_macros.get(option.name, True)
        ]

    return OptionsManager(
        primitive_options=primitive_options,
        macro_options=macro_options
    )
