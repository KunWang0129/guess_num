import os
import json
from typing import Optional, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.num_env import state
from src.num_env.const import NUM_N, REWARD

def _load_sequences(path: str, limit: Optional[int]=None) -> List[str]:
    """Load sequences from JSONL file and convert to strings.

    Each sequence in the file is a list like [0, 4, 1, 6, 4, 1, 0, 6, 8, 8]
    which gets converted to the string "0416410688".

    Args:
        path: Path to the JSONL file containing sequences
        limit: Maximum number of sequences to load (None for all)
    """
    sequences = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Convert list [0,4,1,6,...] to string "0416..." (preserving leading zeros)
            seq_str = ''.join(str(d) for d in data['sequence'])
            sequences.append(seq_str)
            if limit and len(sequences) >= limit:
                break
    return sequences


class NumEnvBase(gym.Env):
    """
    Number sequence guessing environment.

    Actions:
        Can guess any 10-digit sequence in the vocabulary
    State space is defined as:
        * max_turns possibilities for remaining turns
        * Each digit [0-9] has a state of 0/1 for whether it's been guessed before
        * For each digit [0-9] can be in one of 3^NUM_N states: (No, Maybe, Yes)
    Reward:
        Reward is 10 for guessing the right sequence, -10 for failing after max_turns.
    Starting State:
        Random goal sequence
        Initial state with turn 0, all digits Unvisited + Maybe
    """
    def __init__(self, sequences: List[str],
                 max_turns: int,
                 allowable_sequences: Optional[int] = None,
                 frequencies: Optional[List[float]]=None,
                 mask_based_state_updates: bool=False,
                 data_path: Optional[str] = None):
        assert all(len(s) == NUM_N for s in sequences), f'Not all sequences of length {NUM_N}'
        self.sequences = sequences
        self.max_turns = max_turns
        self.allowable_sequences = allowable_sequences
        self.mask_based_state_updates = mask_based_state_updates
        if not self.allowable_sequences:
            self.allowable_sequences = len(self.sequences)

        self.frequencies = None
        if frequencies:
            assert len(sequences) == len(frequencies), f'{len(sequences), len(frequencies)}'
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies)

        self.action_space = spaces.Discrete(len(self.sequences))
        self.observation_space = spaces.MultiDiscrete(state.get_nvec(self.max_turns))

        self.done = True
        self.goal_sequence: int = -1  # Index into self.sequences

        self.state: state.NumState = None
        self.state_updater = state.update
        if self.mask_based_state_updates:
            self.state_updater = state.update_mask

    def step(self, action: int):
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state = self.state_updater(state=self.state,
                                        num_seq=self.sequences[action],
                                        goal_seq=self.sequences[self.goal_sequence])

        reward = 0
        terminated = False
        truncated = False

        if action == self.goal_sequence:
            self.done = True
            terminated = True
            #reward = REWARD
            if state.remaining_steps(self.state) == self.max_turns-1:
                reward = 0#-10*REWARD  # No reward for guessing off the bat
            else:
                #reward = REWARD*(self.state.remaining_steps() + 1) / self.max_turns
                reward = REWARD
        elif state.remaining_steps(self.state) == 0:
            self.done = True
            truncated = True
            reward = -REWARD

        return self.state.copy(), reward, terminated, truncated, {"goal_id": self.goal_sequence}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.state = state.new(self.max_turns)
        self.done = False
        self.goal_sequence = int(np.random.random()*self.allowable_sequences)

        return self.state.copy(), {"goal_id": self.goal_sequence}

    def set_goal_sequence(self, goal_seq: str):
        self.goal_sequence = self.sequences.index(goal_seq)

    def set_goal_id(self, goal_id: int):
        self.goal_sequence = goal_id


class NumEnv10(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 10), max_turns=100)


class NumEnv100(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 100), max_turns=100)


class NumEnv100OneAction(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 100), allowable_sequences=1, max_turns=100)


class NumEnv100WithMask(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 100), max_turns=100,
                         mask_based_state_updates=True)


class NumEnv100TwoAction(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 100), allowable_sequences=2, max_turns=100)


class NumEnv100FullAction(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path), allowable_sequences=100, max_turns=100)


class NumEnv1000(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 1000), max_turns=100)


class NumEnv1000WithMask(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path, 1000), max_turns=100,
                         mask_based_state_updates=True)


class NumEnv1000FullAction(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path), allowable_sequences=1000, max_turns=100)


class NumEnvFull(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path), max_turns=100)


class NumEnvReal(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path), allowable_sequences=2315, max_turns=100)


class NumEnvRealWithMask(NumEnvBase):
    def __init__(self, data_path: str):
        super().__init__(sequences=_load_sequences(data_path), allowable_sequences=2315, max_turns=100,
                         mask_based_state_updates=True)


# Register all environment variants with Gymnasium
gym.register(
    id='NumEnv10-v0',
    entry_point='src.num_env.wordle:NumEnv10',
)

gym.register(
    id='NumEnv100-v0',
    entry_point='src.num_env.wordle:NumEnv100',
)

gym.register(
    id='NumEnv100OneAction-v0',
    entry_point='src.num_env.wordle:NumEnv100OneAction',
)

gym.register(
    id='NumEnv100WithMask-v0',
    entry_point='src.num_env.wordle:NumEnv100WithMask',
)

gym.register(
    id='NumEnv100TwoAction-v0',
    entry_point='src.num_env.wordle:NumEnv100TwoAction',
)

gym.register(
    id='NumEnv100FullAction-v0',
    entry_point='src.num_env.wordle:NumEnv100FullAction',
)

gym.register(
    id='NumEnv1000-v0',
    entry_point='src.num_env.wordle:NumEnv1000',
)

gym.register(
    id='NumEnv1000WithMask-v0',
    entry_point='src.num_env.wordle:NumEnv1000WithMask',
)

gym.register(
    id='NumEnv1000FullAction-v0',
    entry_point='src.num_env.wordle:NumEnv1000FullAction',
)

gym.register(
    id='NumEnvFull-v0',
    entry_point='src.num_env.wordle:NumEnvFull',
)

gym.register(
    id='NumEnvReal-v0',
    entry_point='src.num_env.wordle:NumEnvReal',
)

gym.register(
    id='NumEnvRealWithMask-v0',
    entry_point='src.num_env.wordle:NumEnvRealWithMask',
)
