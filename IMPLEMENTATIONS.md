# Implementation Details

This document provides a detailed summary of the implementation of the RL-based sequence guessing agent. The project is structured into several modules, each responsible for a specific part of the system.

## 1. Environment (`src/environment.py`)

The `src/environment.py` module contains the `SequenceGuessingEnv` class, which is a custom environment for the sequence guessing game. It provides a Gym-like API for the agent to interact with. The reward system has been refactored to separate completion-based rewards from utility-based rewards.

### `SequenceGuessingConfig`

A `dataclass` that holds the configuration for the environment.

- **`sequence_length`**: The length of the sequence to be guessed.
- **`max_episode_steps`**: The maximum number of steps allowed per episode.
- **`num_values`**: The number of possible values for each position in the sequence (e.g., 1-9).
- **`reward_correct_guess`**: The reward given for guessing the entire sequence correctly.
- **`reward_wrong_guess`**: The penalty for making a wrong guess when the sequence is complete.
- **`reward_step`**: The penalty for each step taken.
- **`reward_partial_correct`**: The reward for each correctly guessed position.
- **`provide_feedback`**: A boolean to control whether to provide feedback on partially correct guesses.
- **`allow_repeated_guesses`**: A boolean to control whether the agent is penalized for repeating a guess.
- **`verbose`**: A boolean to enable or disable verbose logging.

### `CompletionReward`

Encapsulates the reward function for correctly guessing the sequence.

- **`__init__(self, reward_correct_guess: float)`**: Initializes the `CompletionReward` class.
- **`reward_correct_guess(self, current_guess: torch.Tensor, target_sequence: torch.Tensor) -> float`**: Returns the reward for a correct guess.

### `UtilityReward`

Encapsulates optional utility-based reward functions, such as a penalty for each step.

- **`__init__(self, reward_name: str, *, reward_step: float)`**: Initializes the `UtilityReward` class.
- **`compute(self) -> float`**: Computes the configured utility reward.

### `SequenceGuessingEnv`

The main class for the sequence guessing environment.

#### `__init__(self, config: Optional[SequenceGuessingConfig] = None, *, sequence_provider: Optional["SequenceGuessingEnv.SequenceProvider"] = None, use_utility_reward: bool = False, utility_reward_name: Optional[str] = None)`

- **Description**: Initializes the `SequenceGuessingEnv`.
- **Inputs**:
    - `config` (Optional[`SequenceGuessingConfig`]): An optional configuration object for the environment.
    - `sequence_provider` (Optional[`SequenceProvider`]): An optional provider for generating sequences.
    - `use_utility_reward` (bool): If `True`, a utility reward (e.g., `reward_step`) is used.
    - `utility_reward_name` (Optional[str]): The name of the utility reward to use.
- **Outputs**: None.

#### `reset(self) -> torch.Tensor`

- **Description**: Resets the environment to start a new episode.
- **Inputs**: None.
- **Outputs**: `torch.Tensor`: The initial state observation.

#### `step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]`

- **Description**: Executes a single step in the environment.
- **Inputs**:
    - `action` (int): The action to be taken.
- **Outputs**: `Tuple[torch.Tensor, float, bool, Dict[str, Any]]`: A tuple containing the next state, reward, done flag, and info dictionary.

#### `_calculate_reward(self) -> float`

- **Description**: Calculates the reward for the current step. It combines the completion reward and any configured utility rewards.
- **Inputs**: None.
- **Outputs**: `float`: The calculated reward.

#### `_is_episode_done(self) -> bool`

- **Description**: Checks if the episode should terminate.
- **Inputs**: None.
- **Outputs**: `bool`: `True` if the episode is done, otherwise `False`.

#### `_get_state(self) -> torch.Tensor`

- **Description**: Constructs and returns the current state representation for the agent. The state no longer includes the target sequence to prevent the agent from learning to copy it. It is composed of the current guess, the last feedback, and metadata.
- **Inputs**: None.
- **Outputs**: `torch.Tensor`: The current state as a tensor.

#### `get_state_with_target(self) -> torch.Tensor`

- **Description**: A new method that returns an augmented state representation that *does* include the true target sequence. This can be used for oracle models or for analysis.
- **Inputs**: None.
- **Outputs**: `torch.Tensor`: The state augmented with the target sequence.

#### `action_to_position_value(self, action: int) -> Tuple[int, int]`

- **Description**: Converts an integer action to a `(position, value)` tuple.
- **Inputs**:
    - `action` (int): The action to convert.
- **Outputs**: `Tuple[int, int]`: The `(position, value)` tuple.

#### `position_value_to_action(self, position: int, value: int) -> int`

- **Description**: Converts a `(position, value)` tuple to an integer action.
- **Inputs**:
    - `position` (int): The position in the sequence.
    - `value` (int): The value to be placed.
- **Outputs**: `int`: The integer action.

#### `get_info(self) -> Dict[str, Any]`

- **Description**: Returns a dictionary with information about the environment's current state and configuration.
- **Inputs**: None.
- **Outputs**: `Dict[str, Any]`: A dictionary containing environment information.

### `create_env_from_config(config_dict: Dict[str, Any]) -> SequenceGuessingEnv`

- **Description**: A factory function that creates an instance of `SequenceGuessingEnv` from a configuration dictionary. **Note**: This function may need to be updated to handle the new `use_utility_reward` and `utility_reward_name` parameters in the `SequenceGuessingEnv` constructor.
- **Inputs**:
    - `config_dict` (Dict[str, Any]): A dictionary containing the environment's configuration parameters.
- **Outputs**: `SequenceGuessingEnv`: A new instance of the environment.

## 2. Model (`src/model.py`)

The `model.py` module defines the neural network architecture for the DQN agent.

- **`SimpleMLP`**: The primary model is a simple Multi-Layer Perceptron (MLP) that takes the state as input and outputs Q-values for all possible actions (both primitive and options).
- **Configuration**: The `ModelConfig` dataclass is used to configure the model's architecture, including the number of hidden layers, activation function, and dropout rate.
- **Hybrid Action Space**: The model's output layer has a size equal to the sum of primitive actions and options, allowing the agent to choose between them.

### `ModelConfig`

A `dataclass` for configuring the neural network models.

- **`__post_init__(self)`**: A method that is automatically called after the `dataclass` is initialized. It sets the `hidden_sizes` to `[128, 128]` if they are not provided in the configuration.

### `SimpleMLP`

The main neural network model, which is a simple Multi-Layer Perceptron (MLP).

#### `__init__(self, config: ModelConfig)`

- **Description**: Initializes the `SimpleMLP` model.
- **Inputs**:
    - `config` (`ModelConfig`): The configuration object for the model.
- **Outputs**: None.

#### `_get_activation(self) -> nn.Module`

- **Description**: A helper method that returns a PyTorch activation function based on the configuration.
- **Inputs**: None.
- **Outputs**: `nn.Module`: A PyTorch activation function (e.g., `nn.ReLU`).

#### `forward(self, state: torch.Tensor) -> torch.Tensor`

- **Description**: Performs a forward pass through the network.
- **Inputs**:
    - `state` (`torch.Tensor`): A tensor representing the current state.
- **Outputs**: `torch.Tensor`: A tensor of Q-values for all actions (primitive and options).

#### `get_primitive_q_values(self, state: torch.Tensor) -> torch.Tensor`

- **Description**: Returns the Q-values for primitive actions only.
- **Inputs**:
    - `state` (`torch.Tensor`): The current state tensor.
- **Outputs**: `torch.Tensor`: A tensor of Q-values for primitive actions.

#### `get_option_q_values(self, state: torch.Tensor) -> torch.Tensor`

- **Description**: Returns the Q-values for options only.
- **Inputs**:
    - `state` (`torch.Tensor`): The current state tensor.
- **Outputs**: `torch.Tensor`: A tensor of Q-values for options.

### `create_model(model_type: str, config: ModelConfig) -> nn.Module`

- **Description**: A factory function that creates and returns a neural network model.
- **Inputs**:
    - `model_type` (str): The type of model to create (currently only "mlp" is supported).
    - `config` (`ModelConfig`): The configuration for the model.
- **Outputs**: `nn.Module`: An initialized neural network model.

### `calculate_state_size(sequence_length: int) -> int`

- **Description**: Calculates the size of the state vector based on the environment's configuration. The formula has been updated to `sequence_length * 2 + 3`. The state now includes the `current_guess`, `last_feedback`, and metadata, but **not** the `target_sequence`.
- **Inputs**:
    - `sequence_length` (int): The length of the sequence to be guessed.
- **Outputs**: `int`: The size of the state vector.

### `calculate_action_space_size(sequence_length: int, num_values: int, num_options: int) -> int`

- **Description**: Calculates the total size of the action space, including both primitive actions and options.
- **Inputs**:
    - `sequence_length` (int): The length of the sequence.
    - `num_values` (int): The number of possible values for each position.
    - `num_options` (int): The number of available options.
- **Outputs**: `int`: The total size of the action space.

## 3. Options Framework (`src/options.py`)

The Options Framework, implemented in `src/options.py`, provides a mechanism for the agent to learn and execute temporally extended actions, known as 'options'. Instead of selecting a single primitive action (guessing a number at a position) at each step, the agent can choose to initiate an option, which then follows its own internal policy for a series of steps.

This hierarchical approach allows the agent to operate at a higher level of abstraction. For example, instead of learning a complex sequence of primitive actions to fill the first half of the sequence, it can learn to simply invoke the `GuessFirstHalfOption`.

The framework consists of:
- A `BaseOption` abstract class defining the common interface for all options.
- A set of concrete 'macro' options that implement specific, high-level strategies (e.g., `ComplementaryGuessOption`).
- `PrimitiveOption`s, which wrap the basic actions of the environment so they can be treated uniformly within the option catalog.
- An `OptionsManager` that maintains a catalog of all available options (both primitive and macro), manages the lifecycle of the active option (initiating, stepping, terminating), and determines which options are available in a given state.

### Helper Functions

- **`_extract_current_guess(state: torch.Tensor, sequence_length: int) -> torch.Tensor`**
  - **Description**: Slices the state tensor to extract the portion representing the current guess of the sequence.
  - **Inputs**:
    - `state` (`torch.Tensor`): The full state tensor.
    - `sequence_length` (`int`): The length of the sequence.
  - **Outputs**: `torch.Tensor`: A tensor containing only the current guess.

- **`_episode_done(state: torch.Tensor) -> bool`**
  - **Description**: Checks the metadata in the state tensor to determine if the current episode has concluded.
  - **Inputs**: `state` (`torch.Tensor`): The full state tensor.
  - **Outputs**: `bool`: `True` if the episode is done, otherwise `False`.

### Core Classes and Enums

- **`OptionStatus(Enum)`**: An enumeration representing the execution status of an option (`ACTIVE`, `TERMINATED`, `INVALID`).

- **`OptionKind(Enum)`**: An enumeration to categorize options as either `PRIMITIVE` (a single action) or `MACRO` (a temporally extended policy).

- **`PrimitiveAction(dataclass)`**: Represents a canonical `(position, value)` guess. Values are 0-indexed.
  - **`to_index(self) -> int`**: Converts the action into a flat integer index used by the environment.

- **`OptionResult(dataclass)`**: A data structure to hold the result of an option's policy execution for a single step, including the action to take, termination status, and metadata.

- **`OptionCatalogEntry(dataclass)`**: Represents a flattened entry in the `OptionsManager` catalog, providing a unified interface for both primitive and macro options.

### `BaseOption` (Abstract Base Class)

The abstract base class for all options, defining the required interface for temporally extended actions. The `can_initiate` method has been removed, meaning any option can be started at any time.

- **`policy(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult`** (abstract)
  - **Description**: Defines the option's internal policy, determining the next primitive action to take.
  - **Inputs**:
    - `state` (`torch.Tensor`): The current environment state.
    - `env_info` (`Dict`): Additional information from the environment.
  - **Outputs**: `OptionResult`: The result of the policy execution.

- **`should_terminate(self, state: torch.Tensor, env_info: Dict[str, Any]) -> bool`** (abstract)
  - **Description**: Checks if the option should terminate in the given state.
  - **Inputs**:
    - `state` (`torch.Tensor`): The current environment state.
    - `env_info` (`Dict`): Additional information from the environment.
  - **Outputs**: `bool`: `True` if the option should terminate, otherwise `False`.

- **`step(self, state: torch.Tensor, env_info: Dict[str, Any]) -> OptionResult`**
  - **Description**: Executes one step of the option, checking for termination conditions and calling its internal policy.
  - **Inputs**:
    - `state` (`torch.Tensor`): The current environment state.
    - `env_info` (`Dict`): Additional information from the environment.
  - **Outputs**: `OptionResult`: The result of the step.

### Concrete Option Implementations

- **`PrimitiveOption(BaseOption)`**: A single-step option that wraps a primitive `(position, value)` guess. It terminates immediately after execution.

- **`GuessFirstHalfOption(BaseOption)`**: A macro-option that restricts its guesses to the first half of the sequence. It terminates when all positions in the first half are filled or the episode ends.

- **`ComplementaryGuessOption(BaseOption)`**: A macro-option that attempts to fill the second half of the sequence with values that are complementary to the first half (summing to 9). It terminates when the episode ends.

- **`OddNumbersOnlyOption(BaseOption)`**: A macro-option that only guesses odd numbers in any available empty position. It terminates when the sequence is full or the episode ends.

- **`GaussianGuessOption(BaseOption)`**: A macro-option that samples guesses from a Gaussian distribution centered at 5, clamping values to the `[0, 9]` range. It terminates when the sequence is full or the episode ends.

### `OptionsManager`

A central manager for the entire options framework.

- **`__init__(self, primitive_options, macro_options)`**: Initializes the manager and builds a unified catalog of all primitive and macro options.

- **Catalog Utilities**:
  - `_build_catalog()`: Creates the unified catalog.
  - `get_catalog()`: Returns the full list of `OptionCatalogEntry` items.
  - `get_entry(index)`: Retrieves a specific entry by its index.
  - `is_primitive(index)`: Checks if an option is primitive.

- **Macro Option Lifecycle**:
  - `get_available_options(state, env_info)`: Returns a list of all available macro option names.
  - `initiate_option(option_name, state, env_info)`: Starts the execution of a specified macro option without any precondition checks.
  - `step_active_option(state, env_info)`: Executes a step of the currently active macro option.
  - `terminate_active_option()`: Forcibly terminates the active macro option.

- **Metadata**:
  - `get_option_stats()`: Returns a dictionary with statistics about option usage.

### Factory Functions

- **`create_default_macro_options() -> List[BaseOption]`**: Creates and returns a list containing instances of all default macro options.

- **`create_primitive_option_catalog(sequence_length: int, num_values: int) -> List[PrimitiveOption]`**: Generates a complete list of all possible `PrimitiveOption`s, with values ranging from 0 to `num_values`.

- **`create_options_manager(...) -> OptionsManager`**: A factory function to construct and configure an `OptionsManager`, allowing for the inclusion of primitive options and selective enabling of macro options.

## 4. Replay Buffer (`src/replay_buffer.py`)

The `replay_buffer.py` module is a critical component for off-policy reinforcement learning algorithms like DQN. Its primary role is to store the agent's experiences—transitions of `(state, action, reward, next_state)`—so the agent can learn from them at a later time. By sampling batches of these experiences randomly, the agent breaks the temporal correlations in the data, leading to more stable and efficient training.

### Key Features & Implementations

-   **`Experience` Dataclass**: A standardized container for a single transition, which includes the core tuple (`state`, `action`, `reward`, `next_state`, `done`) as well as metadata for the Options Framework (`is_option`, `option_name`).
-   **`ReplayBuffer`**: The standard implementation using a `collections.deque`.
    -   **Storage**: Experiences are added via the `.add()` method.
    -   **Sampling**: Provides uniform random sampling through the `.sample()` and `.sample_tensors()` methods.
-   **`PrioritizedReplayBuffer` (PER)**: An advanced buffer that samples more "important" or "surprising" transitions more frequently.
    -   **Priority**: It assigns a priority to each experience, typically based on the TD-error from the learning step.
    -   **Data Structures**: Uses a Sum Tree for efficient, weighted sampling.
    -   **Bias Correction**: Implements importance sampling weights to correct for the bias introduced by non-uniform sampling. The agent can update priorities using the `.update_priorities()` method.
-   **`SegmentedReplayBuffer`**: A specialized buffer that maintains separate storage for different kinds of experiences (e.g., "primitive" vs. "option"). This allows for balanced sampling to ensure rare but important experiences (like those from a specific option) are included in training batches.
-   **Factory Function**: The `create_replay_buffer` function reads the agent's configuration and instantiates the appropriate buffer (`ReplayBuffer` or `PrioritizedReplayBuffer`).

### Usage in the Agent Framework

The replay buffer is a core component managed directly by the `DQNAgent` in `src/agent.py`.

1.  **Initialization**: In the agent's `__init__` method, a replay buffer is created using the `create_replay_buffer` factory based on the configuration provided in `conf/agent/dqn.yaml` (e.g., `prioritized_replay: true`).
2.  **Data Collection**: The main training loop is in `src/train.py`. During each episode, the `DQNAgent.run_episode()` method is called. At every step, after the agent interacts with the environment, it stores the resulting experience by calling `self.replay_buffer.add(...)`.
3.  **Training**: The agent's `train_step()` method is responsible for learning. It first checks if the buffer is ready for sampling (`replay_buffer.can_sample()`). If so, it calls `self.replay_buffer.sample_tensors()` to retrieve a batch of past experiences. These tensors are then used to compute the loss and update the Q-network's weights.

## 5. Agent (`src/agent.py`)

The `src/agent.py` module implements the `DQNAgent`, which is the brain of the reinforcement learning system. It is built on PyTorch Lightning to structure the training process and encapsulates the complete logic for agent-environment interaction, learning, and evaluation.

### Core Components:

-   **`DQNAgent(pl.LightningModule)`**: The central class that orchestrates the agent's behavior. It holds the Q-network and target network, the replay buffer, the options manager, and the optimizer.
-   **`AgentConfig`**: A `dataclass` that centralizes all hyperparameters for the agent, including learning rates, exploration parameters (epsilon), network architecture, and replay buffer settings.

### Walkthrough of Agent Operations:

1.  **Initialization (`__init__`)**:
    -   The agent is initialized with an environment (`SequenceGuessingEnv`) and a configuration object (`AgentConfig`).
    -   It creates the `OptionsManager` if options are enabled.
    -   It instantiates two neural networks: the main `q_network` (for action selection) and a `target_network` (to stabilize learning), using a `ModelConfig` derived from the agent's configuration. The `target_network` is initialized with the same weights as the `q_network`.
    -   A replay buffer is created based on the configuration (standard or prioritized).
    -   An Adam optimizer is set up for training the `q_network`.

2.  **Action Selection (`select_action`)**:
    -   This method implements a sophisticated epsilon-greedy strategy that integrates the Options Framework.
    -   If an option is currently active, the agent executes the next step of that option's internal policy.
    -   If no option is active, the agent decides whether to explore or exploit based on the current `epsilon` value.
        -   **Exploration**: The agent chooses a random action. It has a 50% chance of initiating a random (but available) option versus choosing a random primitive action.
        -   **Exploitation**: The agent uses the `q_network` to predict the Q-values for all possible actions (primitive and options). It then masks any unavailable options and selects the action with the highest Q-value. If the best action is an option, it initiates that option; otherwise, it executes the best primitive action.

3.  **Training (`run_episode` and `train_step`)**:
    -   The `run_episode` method contains the main interaction loop for a single episode.
    -   In each step of the episode, the agent:
        1.  Selects an action using `select_action`.
        2.  Executes the action in the environment to receive the `next_state`, `reward`, and `done` signal.
        3.  Stores this transition `(state, action, reward, next_state, done)` in the replay buffer.
        4.  Calls `train_step` to perform a learning update.
    -   The `train_step` method:
        1.  Samples a batch of experiences from the replay buffer.
        2.  Calculates the current Q-values for the `(state, action)` pairs in the batch.
        3.  Calculates the target Q-values using the Bellman equation: `reward + gamma * max_q(next_state)`. The `target_network` is used to get the Q-values for the `next_state`, which stabilizes training.
        4.  Computes the Mean Squared Error (MSE) loss between the current and target Q-values.
        5.  Performs a standard backpropagation and optimization step to update the `q_network`'s weights.
        6.  Periodically updates the `target_network` using either a hard copy or a soft update, as configured.
        7.  Decays the `epsilon` value to reduce exploration over time.

4.  **Evaluation (`evaluate`)**:
    -   To measure performance, the `evaluate` method runs the agent for a specified number of episodes with exploration turned off (`epsilon = 0.0`).
    -   It computes and returns key metrics like mean reward, mean episode length, and success rate.

5.  **Checkpointing (`save_checkpoint`, `load_checkpoint`)**:
    -   The agent provides methods to save and load its state, including the weights of the networks, the optimizer state, and training progress (like `steps_done` and `epsilon`). This allows for resuming training sessions.

The `DQNAgent` class is the core of the RL agent. It is implemented using PyTorch Lightning.

- **Configuration**: The `AgentConfig` dataclass is used to configure the agent's hyperparameters, such as learning rate, gamma, epsilon for exploration, and target network update frequency.
- **Action Selection**: The agent uses an epsilon-greedy strategy to select actions. It can choose between primitive actions and options. When using options, it has a separate epsilon for option selection.
- **Training**: The agent is trained using the DQN algorithm. It samples experiences from the replay buffer and updates the Q-network to minimize the mean squared error between the predicted Q-values and the target Q-values.
- **Options Integration**: The agent seamlessly integrates the Options Framework. It can select and execute options, and the replay buffer stores option-related information.

### `AgentConfig`

A `dataclass` for configuring the `DQNAgent`.

- **`learning_rate`**: The learning rate for the Adam optimizer.
- **`gamma`**: The discount factor for future rewards.
- **`batch_size`**: The number of experiences to sample from the replay buffer for each training step.
- **`eps_start`**: The initial value of epsilon for the epsilon-greedy exploration strategy.
- **`eps_end`**: The minimum value of epsilon.
- **`eps_decay`**: The decay rate for epsilon.
- **`target_update_freq`**: The frequency (in steps) at which the target network is updated.
- **`soft_update_tau`**: The interpolation parameter for soft target network updates.
- **`use_soft_updates`**: A boolean to control whether to use soft or hard target network updates.
- **`max_episodes`**: The maximum number of episodes to train for.
- **`max_steps_per_episode`**: The maximum number of steps per episode.
- **`use_macro_options`**: A boolean to enable or disable the Options Framework.
- **`option_epsilon`**: A separate epsilon for selecting options.
- **`option_termination_bonus`**: A bonus reward for completing an option.
- **`model_type`**: The type of model to use (currently only "mlp" is supported).
- **`hidden_sizes`**: A list of integers defining the sizes of the hidden layers in the model.
- **`activation`**: The activation function to use in the model.
- **`dropout_rate`**: The dropout rate for the model.
- **`buffer_size`**: The maximum size of the replay buffer.
- **`min_replay_size`**: The minimum number of experiences in the replay buffer before training starts.
- **`prioritized_replay`**: A boolean to enable or disable the Prioritized Replay Buffer.

### `DQNAgent`

The main class for the DQN agent, implemented as a PyTorch Lightning module.

#### `__init__(self, env: SequenceGuessingEnv, config: AgentConfig)`

- **Description**: Initializes the `DQNAgent`.
- **Inputs**:
    - `env` (`SequenceGuessingEnv`): An instance of the sequence guessing environment.
    - `config` (`AgentConfig`): The configuration object for the agent.
- **Outputs**: None.

#### `forward(self, state: torch.Tensor) -> torch.Tensor`

- **Description**: Performs a forward pass through the Q-network.
- **Inputs**:
    - `state` (`torch.Tensor`): The current state tensor.
- **Outputs**: `torch.Tensor`: A tensor of Q-values for all actions.

#### `select_action(self, state: torch.Tensor, epsilon: Optional[float] = None) -> Tuple[int, Dict[str, Any]]`

- **Description**: Selects an action using an epsilon-greedy strategy, with support for the Options Framework.
- **Inputs**:
    - `state` (`torch.Tensor`): The current environment state.
    - `epsilon` (Optional[float]): The epsilon value for exploration. If `None`, the agent's internal epsilon is used.
- **Outputs**: `Tuple[int, Dict[str, Any]]`: A tuple containing the selected action and metadata about the action.

#### `train_step(self) -> Optional[float]`

- **Description**: Performs a single training step, including sampling from the replay buffer, calculating the loss, and updating the Q-network.
- **Inputs**: None.
- **Outputs**: `Optional[float]`: The training loss for the step, or `None` if the replay buffer is not ready for sampling.

#### `_update_target_network(self, hard_update: bool = None)`

- **Description**: Updates the weights of the target network, using either a hard or soft update.
- **Inputs**:
    - `hard_update` (Optional[bool]): If `True`, a hard update is performed. If `None`, the update method is determined by the agent's configuration.
- **Outputs**: None.

#### `run_episode(self, train: bool = True) -> Dict[str, Any]`

- **Description**: Runs a single, complete episode in the environment.
- **Inputs**:
    - `train` (bool): If `True`, the agent will perform training steps during the episode.
- **Outputs**: `Dict[str, Any]`: A dictionary containing statistics about the episode.

#### `evaluate(self, num_episodes: int = 10) -> Dict[str, Any]`

- **Description**: Evaluates the agent's performance over a number of episodes with exploration turned off.
- **Inputs**:
    - `num_episodes` (int): The number of episodes to evaluate.
- **Outputs**: `Dict[str, Any]`: A dictionary containing evaluation statistics.

#### `get_training_stats(self) -> Dict[str, Any]`

- **Description**: Returns a dictionary with comprehensive statistics about the training process.
- **Inputs**: None.
- **Outputs**: `Dict[str, Any]`: A dictionary of training statistics.

#### `save_checkpoint(self, filepath: str) -> None`

- **Description**: Saves the agent's state to a checkpoint file.
- **Inputs**:
    - `filepath` (str): The path to the file where the checkpoint will be saved.
- **Outputs**: None.

#### `load_checkpoint(self, filepath: str) -> None`

- **Description**: Loads the agent's state from a checkpoint file.
- **Inputs**:
    - `filepath` (str): The path to the checkpoint file.
- **Outputs**: None.

## 6. Training (`src/train.py`)

The `src/train.py` script is the main entry point for configuring and running the training process for the RL agent. It uses Hydra for managing complex configurations and orchestrates the entire experiment, from setting up the environment and agent to running the training loop, evaluating performance, and saving results.

### `setup_logging(config: DictConfig) -> None`

-   **Description**: Configures the Python `logging` module for the experiment.
-   **Inputs**:
    -   `config` (`DictConfig`): The Hydra configuration object, which must contain a `logging` section specifying the `level` and `log_dir`.
-   **Outputs**: None.
-   **Functionality**: It sets up a `FileHandler` to save logs to a file (e.g., `training.log`) and a `StreamHandler` to print logs to the console.

### `cleanup_old_checkpoints(checkpoint_dir: str, pattern: str, keep_latest: int = 1) -> None`

-   **Description**: Manages checkpoint files by deleting older ones to save disk space.
-   **Inputs**:
    -   `checkpoint_dir` (str): The directory containing the checkpoint files.
    -   `pattern` (str): A glob pattern (e.g., "best_model_*.ckpt") to identify the relevant checkpoints.
    -   `keep_latest` (int): The number of the most recent checkpoints to preserve.
-   **Outputs**: None.
-   **Functionality**: It finds all files matching the pattern, sorts them by modification time (newest first), and removes all but the `keep_latest` files.

### `create_environment(config: DictConfig) -> SequenceGuessingEnv`

-   **Description**: Factory function that constructs and initializes the `SequenceGuessingEnv`.
-   **Inputs**:
    -   `config` (`DictConfig`): The Hydra configuration containing the `env` section.
-   **Outputs**: `SequenceGuessingEnv`: An initialized environment instance.
-   **Functionality**: It reads the environment parameters from the config, creates a `SequenceGuessingConfig` object, and optionally initializes a sequence data loader (`sequence_provider`) if specified in the configuration.

### `create_agent_config(config: DictConfig) -> AgentConfig`

-   **Description**: Maps the nested Hydra configuration to the flat `AgentConfig` dataclass required by the `DQNAgent`.
-   **Inputs**:
    -   `config` (`DictConfig`): The Hydra configuration containing the `agent` section.
-   **Outputs**: `AgentConfig`: A populated agent configuration object.
-   **Functionality**: This function acts as an adapter, translating the hierarchical structure of the YAML configuration files into the specific format expected by the agent's constructor.

### `train_agent(config: DictConfig) -> DQNAgent`

-   **Description**: The core function that orchestrates the entire training, evaluation, and checkpointing loop.
-   **Inputs**:
    -   `config` (`DictConfig`): The full experiment configuration.
-   **Outputs**: `DQNAgent`: The agent after it has completed training.
-   **Functionality**:
    1.  **Setup**: Sets random seeds, creates the environment, and initializes the `DQNAgent`.
    2.  **Training Loop**: Iterates for `config.training.max_episodes`. In each episode, it calls `agent.run_episode()` to interact with the environment and learn.
    3.  **Logging**: Periodically logs episode statistics (reward, length, epsilon) and option usage.
    4.  **Evaluation**: Periodically runs `agent.evaluate()` to measure performance without exploration.
    5.  **Checkpointing**: Saves the best-performing model based on evaluation scores and also saves periodic checkpoints. It uses `cleanup_old_checkpoints` to manage these files.
    6.  **Early Stopping**: Monitors evaluation performance and stops training if the mean reward does not improve for a configured number of episodes (`early_stopping_patience`).
    7.  **Finalization**: Runs a final, more extensive evaluation and saves the final trained model.

### `run_experiment(config: DictConfig) -> None`

-   **Description**: A wrapper function that manages the complete lifecycle of a single experiment.
-   **Inputs**:
    -   `config` (`DictConfig`): The Hydra configuration for the experiment.
-   **Outputs**: None.
-   **Functionality**: It handles the setup (creating directories, cleaning old checkpoints), calls `train_agent` to run the training, and logs the final results. It includes a `try...except` block to gracefully handle and log any exceptions that occur during the experiment.

### `main(config: DictConfig) -> None`

-   **Description**: The main entry point for the script, decorated with `@hydra.main`.
-   **Inputs**:
    -   `config` (`DictConfig`): The configuration object automatically loaded and populated by Hydra from the `conf/` directory.
-   **Outputs**: None.
-   **Functionality**: It prints the experiment configuration for verification and then calls `run_experiment` to kick off the process.

### `create_demo_script() -> function` and `demo()`

-   **Description**: A pair of functions designed for simple, interactive testing and demonstration.
-   **Inputs**: None.
-   **Outputs**: The `create_demo_script` function returns the `demo` function.
-   **Functionality**: The `demo` function creates a minimal, hard-coded environment and agent configuration, instantiates the `DQNAgent`, and runs a few episodes to demonstrate that the core components are working together correctly. This allows for quick, lightweight checks without needing a full Hydra configuration.

## 7. Data Generation and Loading

### `src/data/sequence_bank.py`

This utility is used to generate datasets of sequences with different properties.

- **Sequence Generators**: It includes several sequence generators, such as `generate_uniform`, `generate_complement_pairs`, `generate_parity_lock`, `generate_gaussian_centered`, and `generate_progression`.
- **Output**: The generated sequences are saved in JSONL format, along with a manifest file containing metadata about the dataset.

### `src/data/loaders.py`

This module provides data loaders for the sequence datasets generated by `sequence_bank.py`.

- **`SequenceBankDataset`**: A class that loads a sequence dataset from a JSONL file into memory.
- **`SequenceBankDataLoader`**: An iterator that provides a stream of sequences from a `SequenceBankDataset`, with options for shuffling and cycling.
- **Integration**: The `create_sequence_loader` function is used in `train.py` to create a sequence loader from the configuration, which is then used to provide target sequences to the environment.
