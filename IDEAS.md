# Project Plan: RL Agent for Sequence Guessing

This document outlines the plan for a new repository to develop a Reinforcement Learning (RL) agent that solves a sequence guessing game. The project will be built with PyTorch Lightning and configured with Hydra.

## 1. The Guessing Game Environment 🎮

The core task is a sequence guessing game where the agent must discover a hidden target sequence, $G$, of length $N$.

-   **Objective**: To correctly guess the entire hidden numerical sequence $G = (g_1, g_2, ..., g_N)$.
-   **State ($s_t$)**: The agent's current version of the sequence, including correctly guessed numbers and empty slots (initialized using uniform distribution as starting state).
-   **Primary Action ($a_t$)**: The agent's action is a tuple $a = (v, i)$, where $v \in \{0, 1, 2, ..., 9\}$ is the value to guess and $i \in \{1, ..., N\}$ is the position (index) in the sequence.
-   **Reward ($R_t$)**: To encourage progress, each correctly placed digit earns a partial reward. Guessing the entire sequence grants an additional completion bonus on that step.

**New: Options** An option defines a temporally extended action, which consist of multiple primary actions. When an option is initiated, there can be a shift in the MDP, where different components might have augmentations or restrictions
- In our setting the options are static and predefined, thus the agent is free to select both primary actions as well as these predefined "options"

When an option is invoked, each action the agent take is still a singular primary action, just following the instructions of the option.

**Option Integration Plan**
- Implement `src/utils/option_utils.py` that enumerates all option functions, exposing the full set of `N × V` primary options that take the current state and return the one-step successor `s_{t+1}`.
- Define composite (non-primary) option functions in the same module so they orchestrate sequences of primary calls and return the resulting `s_{t+k}` for the number of primitive steps executed.
- Reference these option function names from Hydra configs (for example `config/options/*.yaml`) so experiments can enable or disable specific options cleanly.
- Extend the environment wrapper in `src/modules/` to resolve option IDs through `option_utils`, executing primary sequences while still emitting primitive transitions to the base MDP.
- Update the agent's policy head to score both primitive actions and option entries, tag replay transitions with the active option, and add pytest coverage for option execution edge cases alongside logging of option usage in `logs/`.

**Available Options**
Here are all the non-primary options available to the agent:

Option 1: Guess number in the [0:N//2-1] position
- Description: for each index i in [0:N//2-1], guess a number once per index
- Termination: after two guesses for two different indices in [0:N//2-1]

Option 2: Form palindrone
- Description: for existing guesses at position i \in [0:N//2-1], fill position N-i with number such that s_{t+1}[N-i] + s_t[i] = 9 (complement to 9)
- Termination: after filling complements for two indices in [N//2:N-1]

Option 3: Guessing same number block
- Description: guess one number at index i, and then fill either i+1 or i-1 with same number (chooses with equal probability, deterministic if one index go out of bound)
- Termination: inputing two adjacent indices with same number

Option 4: Gaussian distribution
- Description: when guessing for all indices, sample from gaussian distribution centered at 5, variance 2
- Termination: after two guesses

Option 5: Monotonic Sequence Fill
- Description: when guessing for some index i, sample uniformly from numbers larger than number at index i-1 (do not guess at i = 0)
- Termination: after two guesses

Option 6: Sum Targeter
- Description: when guessing for some index i, fill the number such that the sum of all numbers is closest to 60
- Termination: after two guesses

Option 7: High-Low Alternator
- Description: when guessing for some index i, if i is even only guess from {0,1,2}, if i is odd only guess from {7,8,9}
- Termination: after two guesses

Option 8: Repetitive pattern
- Description: for a fixed pattern "84", fill in adjacent indices according to the pattern
- Termination: after two primary actions

Option 9: Digits of Pi
- Description: using first N digits of pi (3141592653...) to fill in for index i
- Termination: after two primary actions

Option 10: Difference maintaining
- Description: for each new guess at i, make sure the new guess is within difference of 2 to both min and max of the sequence digits
- Termination: after two primary actions
---

## 2. Data Generation

The objective of generating data for this environment is very simple: generate number sequence of length N. 
The basic generation steps for a mixed dataset:
- First generate based on **Uniform Distribution** as default
- Then transform each sequence according to starting condition of each special functions
- Record the number sequences transformed by special functions and save as metadata
Here are the different configurations to follow:

**Uniform Distribution** (unif)
- Description: For each index, sample uniformly from vocabulary/digits
- Starting condition: None

**Palindrome**
- Description: A number sequence where digits are symmetric around the center (s[i] = s[N-1-i] for all valid i)
- Starting condition: When the first and last digit both equal 0

**Same Number Block**
- Description: A number sequence where adjacent indices contain the same digit values, forming contiguous blocks of identical numbers
- Starting condition: When first and second number both equal 1

**Gaussian Sequence**
- Description: A number sequence where each digit is sampled from a Gaussian distribution centered at 5 with variance 2, clipped to [0,9]
- Starting condition: When the first and last number both equal 5

**Monotonic Sequence**
- Description: A number sequence that monotonically increases with index
- Starting condition: When first number = 0 and last number = 9

**Sum Targeter**
- Description: A number sequence where all digits sum to a target value of 60 (or closest achievable sum for given sequence length)
- Starting condition: When first number and last number both equal 6

**High-Low Alternator**
- Description: A number sequence with alternating low and high digits: even indices contain {0,1,2}, odd indices contain {7,8,9}
- Starting condition: When first number = 1 and last number = 8

**Repetitive Pattern**
- Description: A number sequence formed entirely by repeating the pattern "84" cyclically across all positions
- Starting condition: When first number = 8 and last number = 4

**Digits of Pi**
- Description: A number sequence formed by the first N digits of π (3.141592653...), excluding the decimal point
- Starting condition: When first two numbers form "31"

**Difference Maintaining**
- Description: A number sequence where every digit stays within a range of 2 from both the minimum and maximum values in the sequence
- Starting condition: When first and last number's difference is exactly 4



## 3. Module Restructure Plan

Goal: reorganize `src/module` into focused subpackages for the Deep Q Network stack (`dqn/`) and the number-sequence environment (`num_seq/`) while keeping existing behaviour intact.

### Discovery & Scoping
- Inventory the current contents of `src/module/` to map each class or helper to either the DQN agent stack or the number-sequence environment concerns.
- Identify shared utilities (if any) that should remain at `src/module/__init__.py` or move to a dedicated `src/module/utils.py` so neither subpackage depends on the other.

### Directory Layout Changes
- Create `src/module/dqn/` with an `__init__.py` that re-exports the public agent, model, and replay APIs for compatibility.
- Create `src/module/num_seq/` with an `__init__.py` that exposes environment, options, and rewards constructs.
- Migrate or recreate the following modules under the new structure:
  - `src/module/dqn/agent.py`: wrap the existing Lightning agent logic so it receives the environment interface and Q-network, sampling options as actions.
  - `src/module/dqn/model.py`: split into `StateEmbedding` (encodes observations) and `QValueNetwork` (3-layer MLP returning logits passed through `softmax`). Preserve weight initialisation and device management.
  - `src/module/dqn/experience.py`: move the replay buffer implementation largely unchanged, updating imports only.
  - `src/module/num_seq/option.py`: hold primary and macro option definitions; ensure signatures and identifiers remain stable.
  - `src/module/num_seq/rewards.py`: define completion and utility reward helpers invoked by the environment.
  - `src/module/num_seq/environment.py`: encapsulate the sequence guessing environment, depending on `option` and `rewards` modules instead of mixed imports.

### Import & Config Updates
- Search the repository (`rg "module." src conf`) to locate all import sites and update them to the new package paths, keeping Hydra configuration entries in sync.
- Adjust any `__all__`, dataclass type hints, or Lightning module references that rely on old module paths to avoid circular imports.
- Verify Hydra configs (`conf/agent`, `conf/env`) refer to the relocated classes (e.g., `module_path: module.dqn.agent.DQNAgent`).

### Backwards Compatibility & Clean-up
- Provide convenience re-exports in `src/module/__init__.py` so existing notebooks or scripts importing from `module import Agent` continue to work until consumers migrate.
- Remove obsolete files or empty directories in `src/module/` after the split and ensure no duplicates remain.
- Update documentation (`README.md`, `IMPLEMENTATIONS.md`) if they reference old layouts.

### Validation & Follow-up
- Run `uv run python -m src.train trainer.fast_dev_run=true` to confirm training still composes the agent, network, and environment correctly post-move.
- Add TODOs or future work notes if further refactors (e.g., shared base classes) are identified during the migration.
