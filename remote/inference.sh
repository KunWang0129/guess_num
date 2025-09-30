#!/bin/bash

# Inference script for guess_num project

echo "Running inference..."
uv run -m src.infer
uv run python -m src.infer env.target_sequence="[8,4,8,4,8,4,8,4,8,4]" --- IGNORE ---

# View composed configuration
# uv run python -m src.train --cfg job