#!/bin/bash

# Inference script for guess_num project

echo "Running inference..."
uv run -m src.infer_dqn

# View composed configuration
# uv run python -m src.train --cfg job