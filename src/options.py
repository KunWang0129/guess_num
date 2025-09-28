"""
Options Framework Implementation

This module implements the Options Framework for the sequence guessing game.
Options define temporally extended actions that consist of multiple primary actions.
Each option provides a policy for selecting actions while it's active.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random
from enum import Enum

import torch


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
        if not 1 <= self.value <= self.num_values:
            raise ValueError(
                f"PrimitiveAction value {self.value} out of bounds for num_values={self.num_values}"
            )

    def to_index(self) -> int:
        """Convert primitive action to flat index used by the environment"""
        # Environment encodes values as 1-indexed, so shift back to 0-index here
        return self.position * self.num_values + (self.value - 1)


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
            description="Only guess numbers in the first half of sequence positions"
        )
        self.max_steps = 20  # Prevent infinite loops


    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """Terminate when all first half positions are filled or episode is done"""
        sequence_length = env_info.get('sequence_length', 4)
        current_guess = _extract_current_guess(state, sequence_length)
        first_half_end = sequence_length // 2
        episode_done = _episode_done(state)  # episode_done is second to last element

        # Terminate if all first half positions are filled or episode is done
        return all(current_guess[pos].item() > 0 for pos in range(first_half_end)) or episode_done

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """Choose a random valid position-value pair in the first half"""
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)
        first_half_end = sequence_length // 2

        # Find empty positions in first half
        empty_positions = [pos for pos in range(first_half_end) if current_guess[pos].item() == 0]

        if not empty_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No empty positions in first half"}
            )

        # Choose random position and value
        position = random.choice(empty_positions)
        value = random.randint(1, num_values)

        primitive_action = PrimitiveAction(position, value, num_values)

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=False,
            status=OptionStatus.ACTIVE,
            metadata={
                "position": position,
                "value": value,
                "available_positions": empty_positions
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
            name="complementary_guess",
            description="Fill complementary positions to make pairs sum to 9"
        )
        self.max_steps = 10


    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """Terminate when episode is done"""
        episode_done = _episode_done(state)
        return episode_done

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
            current_value = current_guess[i].item()
            complement_value = current_guess[complement_pos].item()
            if current_value > 0 and complement_value == 0:
                needed_value = 9 - current_value
                if 1 <= needed_value <= num_values:
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

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=False,
            status=OptionStatus.ACTIVE,
            metadata={
                "position": position,
                "value": value,
                "complement_of": sequence_length - 1 - position,
                "original_value": current_guess[sequence_length - 1 - position],
                "valid_pairs": valid_pairs
            }
        )


class OddNumbersOnlyOption(BaseOption):
    """
    Option that only guesses odd numbers (1, 3, 5, 7, 9) in any empty position.
    """

    def __init__(self):
        super().__init__(
            name="odd_numbers_only",
            description="Only guess odd numbers in any position"
        )
        self.max_steps = 15


    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """Terminate when no empty positions remain or episode done"""
        sequence_length = env_info.get('sequence_length', 4)
        current_guess = _extract_current_guess(state, sequence_length)
        episode_done = _episode_done(state)
        return all(current_guess[pos].item() > 0 for pos in range(sequence_length)) or episode_done

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """Choose random empty position and random odd number"""
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)

        # Find empty positions
        empty_positions = [pos for pos in range(sequence_length) if current_guess[pos].item() == 0]

        if not empty_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No empty positions available"}
            )

        # Get odd numbers within range
        odd_numbers = [i for i in range(1, num_values + 1, 2)]  # 1, 3, 5, 7, 9

        # Choose random position and odd value
        position = random.choice(empty_positions)
        value = random.choice(odd_numbers)

        primitive_action = PrimitiveAction(position, value, num_values)

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=False,
            status=OptionStatus.ACTIVE,
            metadata={
                "position": position,
                "value": value,
                "available_odd_numbers": odd_numbers,
                "empty_positions": empty_positions
            }
        )


class GaussianGuessOption(BaseOption):
    """
    Option that guesses numbers based on a Gaussian distribution centered at 5.
    Values are sampled from N(5, 1.5) and clamped to [0, 9] range.
    """

    def __init__(self, mean: float = 5.0, std: float = 1.5):
        super().__init__(
            name="gaussian_guess",
            description=f"Guess numbers from Gaussian distribution N({mean}, {std})"
        )
        self.mean = mean
        self.std = std
        self.max_steps = 20

    def should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool:
        """Terminate when no empty positions remain or episode done"""
        sequence_length = env_info.get('sequence_length', 4)
        current_guess = _extract_current_guess(state, sequence_length)
        episode_done = _episode_done(state)
        return all(current_guess[pos].item() > 0 for pos in range(sequence_length)) or episode_done

    def policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult:
        """Choose random empty position and Gaussian-sampled value"""
        sequence_length = env_info.get('sequence_length', 4)
        num_values = env_info.get('num_values', 9)
        current_guess = _extract_current_guess(state, sequence_length)

        # Find empty positions
        empty_positions = [pos for pos in range(sequence_length) if current_guess[pos].item() == 0]

        if not empty_positions:
            return OptionResult(
                action=None,
                should_terminate=True,
                status=OptionStatus.TERMINATED,
                metadata={"reason": "No empty positions available"}
            )

        # Sample value from Gaussian and clamp to valid range
        raw_value = random.gauss(self.mean, self.std)
        value = int(round(raw_value))
        value = max(1, min(num_values, value))

        # Choose random position
        position = random.choice(empty_positions)

        primitive_action = PrimitiveAction(position, value, num_values)

        return OptionResult(
            action=primitive_action.to_index(),
            primitive_action=primitive_action,
            should_terminate=False,
            status=OptionStatus.ACTIVE,
            metadata={
                "position": position,
                "value": value,
                "raw_sampled_value": raw_value,
                "gaussian_params": {"mean": self.mean, "std": self.std},
                "empty_positions": empty_positions
            }
        )


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
        OddNumbersOnlyOption(),
        GaussianGuessOption()
    ]


def create_default_options() -> List[BaseOption]:
    """Backward-compatible alias for default macro options"""
    return create_default_macro_options()


def create_primitive_option_catalog(sequence_length: int, num_values: int) -> List[PrimitiveOption]:
    """Enumerate the full `(position, value)` primitive catalog"""
    primitive_options: List[PrimitiveOption] = []
    for position in range(sequence_length):
        for value in range(1, num_values + 1):
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
