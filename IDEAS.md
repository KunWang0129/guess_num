# Project Plan: RL Agent for Sequence Guessing

This document outlines the plan for a new repository to develop a Reinforcement Learning (RL) agent that solves a sequence guessing game. The project will be built with PyTorch Lightning and configured with Hydra.

## 1. The Guessing Game Environment 🎮

The core task is a sequence guessing game where the agent must discover a hidden target sequence, $G$, of length $N$.

-   **Objective**: To correctly guess the entire hidden numerical sequence $G = (g_1, g_2, ..., g_N)$.
-   **State ($s_t$)**: The agent's current version of the sequence, including correctly guessed numbers and empty slots.
-   **Primary Action ($a_t$)**: The agent's action is a tuple $a = (v, i)$, where $v \in \{1, ..., 9\}$ is the value to guess and $i \in \{1, ..., N\}$ is the position (index) in the sequence.
-   **Reward ($R_t$)**: We will start with a sparse reward setting. A reward of **+1** is given only when the entire sequence is guessed correctly. All other states result in a reward of **0**.

**New: Options** An option defines a temporally extended action, which consist of multiple primary actions. When an option is initiated, there can be a shift in the MDP, where different components might have augmentations or restrictions
- In our setting the options are static and predefined, thus the agent is free to select both primary actions as well as these predefined "options"
To illustrate, here are some examples of options:
- Option 1: Guess numbers only in the [0:N//2-1] positions
- Option 2: For existing guesses at position i \in [0:N//2-1], fill position N-i with number such that s_{t+1}[N-i] + s_t[i] = 10 (complement to 10)
- Option 3: Only guess odd numbers
- Option 4: Guess numbers based on Gaussian distribution centered at 5

When an option is invoked, each action the agent take is still a singular primary action, just following the instructions of the option.

**Option Integration Plan**
- Implement `src/utils/option_utils.py` that enumerates all option functions, exposing the full set of `N × V` primary options that take the current state and return the one-step successor `s_{t+1}`.
- Define composite (non-primary) option functions in the same module so they orchestrate sequences of primary calls and return the resulting `s_{t+k}` for the number of primitive steps executed.
- Reference these option function names from Hydra configs (for example `config/options/*.yaml`) so experiments can enable or disable specific options cleanly.
- Extend the environment wrapper in `src/modules/` to resolve option IDs through `option_utils`, executing primary sequences while still emitting primitive transitions to the base MDP.
- Update the agent's policy head to score both primitive actions and option entries, tag replay transitions with the active option, and add pytest coverage for option execution edge cases alongside logging of option usage in `logs/`.
---