#!/bin/bash

# Training script for guess_num project

# # Fast development run (smoke test)
# echo "Running fast development training..."
# uv run python -m src.train trainer.fast_dev_run=true

# Full training run
echo "Running full training..."
uv run python -m src.train_dqn

# Training with custom parameters example
# uv run python -m src.train agent.learning_rate=1e-4 agent.max_episodes=2000
# uv run python -m src.train env.sequence_length=6 env.num_values=10
# uv run python -m src.train agent.use_macro_options=false