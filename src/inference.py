"""Simplified inference entry point for the sequence guessing agent."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .agent import AgentConfig, DQNAgent
from .data import SequenceBankDataLoader, create_sequence_loader
from .environment import SequenceGuessingConfig, SequenceGuessingEnv


log = logging.getLogger(__name__)


@dataclass
class SimpleInferenceResult:
    """Summary of the simplified inference run."""

    success_rate: float


def setup_device(device_spec: str) -> torch.device:
    """Select inference device based on specification."""
    if device_spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("Using GPU: %s", torch.cuda.get_device_name())
        else:
            device = torch.device("cpu")
            log.info("Using CPU")
    else:
        device = torch.device(device_spec)
        log.info("Using specified device: %s", device)

    return device


def load_agent_config_from_checkpoint(checkpoint_path: str) -> AgentConfig:
    """Load the agent configuration stored inside a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Use weights_only=False to allow loading custom classes like AgentConfig
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")

    if config is None:
        raise ValueError("Checkpoint does not contain agent configuration")

    if not isinstance(config, AgentConfig):
        if isinstance(config, dict):
            config = AgentConfig(**config)
        else:
            raise ValueError("Invalid agent configuration in checkpoint")

    log.info("Loaded checkpoint from %s", checkpoint_path)
    return config


def instantiate_env_config(env_cfg: Optional[DictConfig]) -> SequenceGuessingConfig:
    """Create :class:`SequenceGuessingConfig` from Hydra configuration."""
    env_dict: Dict[str, Any] = {}
    if env_cfg is not None:
        env_dict = dict(OmegaConf.to_container(env_cfg, resolve=True))
    return SequenceGuessingConfig(**env_dict)


def apply_env_overrides(
    env_config: SequenceGuessingConfig,
    overrides_cfg: Optional[DictConfig],
) -> None:
    """Apply optional overrides onto the base environment configuration."""
    if overrides_cfg is None or not overrides_cfg.get("enabled", False):
        return

    overrides = dict(OmegaConf.to_container(overrides_cfg, resolve=True))
    overrides.pop("enabled", None)

    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(env_config, key):
            setattr(env_config, key, value)
        else:
            log.warning("Ignoring unknown environment override '%s'", key)


def initialize_sequence_loader(
    data_cfg: Optional[DictConfig],
    *,
    sequence_length: int,
    seed: int,
) -> SequenceBankDataLoader:
    """Instantiate dataset-backed sequence loader shared with training."""
    if data_cfg is None or not data_cfg.get("enabled", False):
        raise ValueError("Enable `data_loader.enabled` to run inference with dataset-backed sequences")

    cfg_dict = dict(OmegaConf.to_container(data_cfg, resolve=True))
    cfg_dict.setdefault("seed", seed)

    loader = create_sequence_loader(cfg_dict, sequence_length=sequence_length)
    summary = loader.dataset.summary()
    log.info(
        "Loaded sequence dataset %s (size=%d, families=%s)",
        summary.get("path"),
        summary.get("size"),
        summary.get("families"),
    )
    return loader


def create_agent(
    checkpoint_path: str,
    agent_config: AgentConfig,
    env: SequenceGuessingEnv,
    device: torch.device,
    *,
    deterministic: bool,
) -> DQNAgent:
    """Instantiate agent, load weights, and switch to evaluation mode."""
    agent = DQNAgent(env, agent_config)
    agent.to(device)
    agent.load_checkpoint(checkpoint_path)
    agent.eval()
    agent.epsilon = 0.0 if deterministic else agent_config.eps_end
    return agent


def extract_final_guess(final_state: Any, sequence_length: int) -> Tuple[int, ...]:
    """Return the final guess portion of the agent state as a tuple."""
    if isinstance(final_state, torch.Tensor):
        tensor = final_state[:sequence_length].to(dtype=torch.int64)
        return tuple(tensor.tolist())

    guess_values = list(final_state[:sequence_length])
    return tuple(int(value) for value in guess_values)


def run_dataset_inference(
    agent: DQNAgent,
    env: SequenceGuessingEnv,
    num_episodes: int,
) -> SimpleInferenceResult:
    """Evaluate agent success rate across a fixed number of sequences."""
    successes = 0

    for _ in range(num_episodes):
        with torch.no_grad():
            episode_stats = agent.run_episode(train=False)

        target_tensor = getattr(env, "target_sequence", None)
        if target_tensor is None:
            log.warning("Environment missing target sequence; counting as failure")
            continue

        final_state = episode_stats.get("final_state")
        if final_state is None:
            log.warning("Episode result missing final state; counting as failure")
            continue

        sequence_length = env.config.sequence_length
        final_guess = extract_final_guess(final_state, sequence_length)
        target_sequence = tuple(int(x) for x in target_tensor.tolist())

        if final_guess == target_sequence:
            successes += 1

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    return SimpleInferenceResult(success_rate=success_rate)


@hydra.main(version_base=None, config_path="../conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Simplified Hydra entry point for inference."""
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Running simplified inference pipeline")
    log.debug("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = setup_device(cfg.device)
    agent_config = load_agent_config_from_checkpoint(cfg.checkpoint.path)

    evaluation_cfg = cfg.get("evaluation")
    seed = cfg.get("seed", 0)
    if evaluation_cfg is not None:
        seed = evaluation_cfg.get("seed", seed)
    deterministic = evaluation_cfg.get("deterministic", True) if evaluation_cfg else True

    env_cfg = cfg.get("env")
    if env_cfg is None:
        raise ValueError("Inference configuration must include an environment profile")

    env_config = instantiate_env_config(env_cfg)
    apply_env_overrides(env_config, cfg.get("env_overrides"))

    sequence_loader = initialize_sequence_loader(
        cfg.get("data_loader"),
        sequence_length=env_config.sequence_length,
        seed=seed,
    )

    env = SequenceGuessingEnv(env_config, sequence_provider=sequence_loader)

    agent_config.max_steps_per_episode = env.config.max_episode_steps
    agent = create_agent(
        cfg.checkpoint.path,
        agent_config,
        env,
        device,
        deterministic=deterministic,
    )

    dataset_size = len(sequence_loader)
    requested_raw = evaluation_cfg.get("num_episodes") if evaluation_cfg else None
    requested_episodes = int(requested_raw) if requested_raw is not None else None
    if requested_episodes is None or requested_episodes <= 0:
        episodes_to_run = dataset_size
    else:
        episodes_to_run = min(dataset_size, requested_episodes)
        if requested_episodes > dataset_size:
            log.warning(
                "Requested %d evaluation episodes but dataset only has %d; limiting to dataset size",
                requested_episodes,
                dataset_size,
            )

    result = run_dataset_inference(agent, env, episodes_to_run)

    log.info(
        "Evaluated %d sequences (dataset_size=%d, sequence_length=%d, max_episode_steps=%d)",
        episodes_to_run,
        dataset_size,
        env.config.sequence_length,
        env.config.max_episode_steps,
    )
    log.info("Success rate: %.2f%%", result.success_rate * 100.0)


if __name__ == "__main__":
    main()
