# Guess Number RL Decomposition

This project implements a Deep Reinforcement Learning agent to solve a sequence guessing game. The agent uses the Options Framework to decompose the problem into a hierarchy of actions, including both primitive actions and temporally extended options.

This project is built with PyTorch Lightning for the training structure and Hydra for configuration management.

## Features

- **Deep Q-Network (DQN) Agent**: The core of the agent is a DQN that learns to solve the sequence guessing game.
- **Options Framework**: The agent can use a set of predefined options to perform temporally extended actions, allowing for more complex behaviors.
- **Configurability**: The project is highly configurable using Hydra. You can easily change the environment, agent, and training parameters without modifying the code.
- **Reproducibility**: The use of PyTorch Lightning and Hydra ensures that experiments are reproducible.
- **Data Generation**: The project includes a tool to generate datasets of sequences with different characteristics.

## Project Structure

```
├── conf/                           # Configuration directory
│   ├── agent/
│   │   └── dqn.yaml               # DQN agent configurations
│   ├── env/
│   │   └── sequence_guesser.yaml  # Environment configurations
│   ├── trainer/
│   │   └── default.yaml           # PyTorch Lightning trainer configs
│   └── config.yaml                # Main configuration file
└── src/                           # Source code directory
    ├── environment.py             # Guessing game environment
    ├── agent.py                   # Core LightningModule RL agent
    ├── model.py                   # Neural network architectures
    ├── options.py                 # Options framework implementation
    ├── replay_buffer.py           # Experience replay buffer
    └── train.py                   # Main training entry point
```

## Getting Started

### Prerequisites

- Python 3.8+
- `uv` package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/guess_num.git
   cd guess_num
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

### Running the Training

To start the training with the default configuration, run the following command from the root of the project:

```bash
uv run python -m src.train
```

### Generating Datasets

To generate a dataset of sequences, you can use the `sequence_bank.py` script. The configuration for the dataset generation is in `conf/data/sequence_bank.yaml`.

```bash
uv run python -m src.data.sequence_bank
```

## Configuration

The project uses Hydra for configuration management. The main configuration file is `conf/config.yaml`, which composes the final configuration from the `agent`, `env`, and `trainer` sub-directories.

You can override any configuration parameter from the command line. For example, to change the learning rate and the number of episodes, you can run:

```bash
uv run python -m src.train agent.learning_rate=1e-4 agent.max_episodes=2000
```

## Usage

Here are some examples of how to run the training with different configurations:

- **Run a fast development run (for smoke testing):**

  ```bash
  uv run python -m src.train trainer.fast_dev_run=true
  ```

- **Change the sequence length and number of values:**

  ```bash
  uv run python -m src.train env.sequence_length=6 env.num_values=10
  ```

- **Disable options and use only primitive actions:**

  ```bash
  uv run python -m src.train agent.use_macro_options=false
  ```

## Testing

Automated tests are not yet established. When they are, they will be located in the `tests/` directory and can be run with `pytest`.

## Contributing

Contributions are welcome! Please read the [Repository Guidelines](AGENTS.md) for more information on how to contribute to this project.
