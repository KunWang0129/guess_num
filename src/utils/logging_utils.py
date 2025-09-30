"""Helper functions for logging artifacts and summaries."""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf


def write_option_frequencies_csv(path: str, counts: Mapping[str, int]) -> None:
    """Persist option usage counts along with normalised frequencies."""
    if not counts:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = sum(max(int(value), 0) for value in counts.values())

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["option_name", "count", "frequency"])

        if total == 0:
            for name in sorted(counts):
                writer.writerow([name, int(counts[name]), 0.0])
            return

        for name in sorted(counts):
            count = max(int(counts[name]), 0)
            frequency = count / total if total else 0.0
            writer.writerow([name, count, frequency])


def append_option_frequencies_timeseries(
    path: str,
    episode: int,
    cumulative_counts: Mapping[str, int]
) -> None:
    """Append cumulative option frequencies at a given episode to a time-series CSV.

    Args:
        path: Path to the CSV file
        episode: Current episode number
        cumulative_counts: Cumulative option counts up to this episode
    """
    if not cumulative_counts:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Check if file exists to determine if we need to write header
    file_exists = os.path.isfile(path)
    total = sum(max(int(value), 0) for value in cumulative_counts.values())

    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)

        # Write header if file is new
        if not file_exists:
            option_names = sorted(cumulative_counts.keys())
            header = ["episode"] + [f"{name}_count" for name in option_names] + [f"{name}_freq" for name in option_names]
            writer.writerow(header)

        # Write data row
        option_names = sorted(cumulative_counts.keys())
        row = [episode]

        # Add counts
        for name in option_names:
            count = max(int(cumulative_counts[name]), 0)
            row.append(count)

        # Add frequencies
        for name in option_names:
            count = max(int(cumulative_counts[name]), 0)
            frequency = count / total if total > 0 else 0.0
            row.append(frequency)

        writer.writerow(row)


def init_wandb_run(config: DictConfig, *, logger: Optional[logging.Logger] = None, **wandb_kwargs: Any):
    """Initialise a Weights & Biases run when enabled in the logging configuration."""
    use_wandb = getattr(config.logging, "wandb", False)
    if not use_wandb:
        return None

    try:
        import wandb
    except ImportError:  # pragma: no cover - defensive guard when wandb missing
        log = logger or logging.getLogger(__name__)
        log.warning("W&B logging requested but the `wandb` package is not installed.")
        return None

    wandb_config = OmegaConf.to_container(config, resolve=True)
    return wandb.init(
        project=config.project.name,
        name=config.experiment.name,
        config=wandb_config,
        reinit=True,
        **wandb_kwargs,
    )
