"""
Keep the state in a 1D int array

index[0] = remaining steps
Rest of data is laid out as binary array

[1..27] = whether char has been guessed or not

[[status, status, status, status, status]
 for _ in "ABCD..."]
where status has codes
 [1, 0, 0] - char is definitely not in this spot
 [0, 1, 0] - char is maybe in this spot
 [0, 0, 1] - char is definitely in this spot
"""
import collections
from typing import List
import numpy as np

from src.num_env.const import NUM_LIST, NUM_N


NumState = np.ndarray


def get_nvec(max_turns: int):
    return [max_turns] + [2] * len(NUM_LIST) + [2] * 3 * NUM_N * len(NUM_LIST)


def new(max_turns: int) -> NumState:
    return np.array(
        [max_turns] + [0] * len(NUM_LIST) + [0, 1, 0] * NUM_N * len(NUM_LIST),
        dtype=np.int32)


def remaining_steps(state: NumState) -> int:
    return state[0]


NO = 0
SOMEWHERE = 1
YES = 2


def update_from_mask(state: NumState, num_seq: int, mask: List[int]) -> NumState:
    """
    return a copy of state that has been updated to new state

    From a mask we need slighty different logic since we don't know the
    goal number.

    :param state:
    :param num_seq:
    :param mask:
    :return:
    """
    state = state.copy()

    # Convert num_seq to list of digits
    num_str = str(num_seq)
    digits = [int(d) for d in num_str]

    prior_yes = []
    prior_maybe = []
    # We need two passes because first pass sets definitely yesses
    # second pass sets the no's for those who aren't already yes
    state[0] -= 1
    for i, digit in enumerate(digits):
        digit_idx = NUM_LIST.index(digit)
        offset = 1 + len(NUM_LIST) + digit_idx * NUM_N * 3
        state[1 + digit_idx] = 1
        if mask[i] == YES:
            prior_yes.append(digit)
            # digit at position i = yes, all other digits at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for other_digit_idx in range(len(NUM_LIST)):
                if other_digit_idx != digit_idx:
                    oc_offset = 1 + len(NUM_LIST) + other_digit_idx * NUM_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]

    for i, digit in enumerate(digits):
        digit_idx = NUM_LIST.index(digit)
        offset = 1 + len(NUM_LIST) + digit_idx * NUM_N * 3
        if mask[i] == SOMEWHERE:
            prior_maybe.append(digit)
            # Digit at position i = no, other digits stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        elif mask[i] == NO:
            # Need to check this first in case there's prior maybe + yes
            if digit in prior_maybe:
                # Then the maybe could be anywhere except here
                state[offset+3*i:offset+3*i+3] = [1, 0, 0]
            elif digit in prior_yes:
                # No maybe, definitely a yes, so it's zero everywhere except the yesses
                for j in range(NUM_N):
                    # Only flip no if previously was maybe
                    if state[offset + 3 * j:offset + 3 * j + 3][1] == 1:
                        state[offset + 3 * j:offset + 3 * j + 3] = [1, 0, 0]
            else:
                # Just straight up no
                state[offset:offset+3*NUM_N] = [1, 0, 0]*NUM_N

    return state


def get_mask(num_seq: int, goal_seq: int) -> List[int]:
    # Definite yesses first
    num_str = str(num_seq)
    goal_str = str(goal_seq)
    digits = [int(d) for d in num_str]
    goal_digits = [int(d) for d in goal_str]

    mask = [0] * len(digits)
    counts = collections.Counter(goal_digits)
    for i, digit in enumerate(digits):
        if goal_digits[i] == digit:
            mask[i] = 2
            counts[digit] -= 1

    for i, digit in enumerate(digits):
        if mask[i] == 2:
            continue
        elif digit in counts:
            if counts[digit] > 0:
                mask[i] = 1
                counts[digit] -= 1
            else:
                for j in range(i+1, len(mask)):
                    if mask[j] == 2:
                        continue
                    mask[j] = 0

    return mask

def update_mask(state: NumState, num_seq: int, goal_seq: int) -> NumState:
    """
    return a copy of state that has been updated to new state

    :param state:
    :param num_seq:
    :param goal_seq:
    :return:
    """
    mask = get_mask(num_seq, goal_seq)
    return update_from_mask(state, num_seq, mask)


def update(state: NumState, num_seq: int, goal_seq: int) -> NumState:
    state = state.copy()

    # Convert to digit lists
    num_str = str(num_seq)
    goal_str = str(goal_seq)
    digits = [int(d) for d in num_str]
    goal_digits = [int(d) for d in goal_str]

    state[0] -= 1
    for i, digit in enumerate(digits):
        digit_idx = NUM_LIST.index(digit)
        offset = 1 + len(NUM_LIST) + digit_idx * NUM_N * 3
        state[1 + digit_idx] = 1
        if goal_digits[i] == digit:
            # digit at position i = yes, all other digits at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for other_digit_idx in range(len(NUM_LIST)):
                if other_digit_idx != digit_idx:
                    oc_offset = 1 + len(NUM_LIST) + other_digit_idx * NUM_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]
        elif digit in goal_digits:
            # Digit at position i = no, other digits stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        else:
            # Digit at all positions = no
            state[offset:offset + 3 * NUM_N] = [1, 0, 0] * NUM_N

    return state

