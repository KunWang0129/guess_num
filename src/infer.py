"""Manual inference entry point for the sequence guessing agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .test import (
    apply_env_overrides,
    create_agent,
    instantiate_env_config,
    load_agent_config_from_checkpoint,
    setup_device,
)
from .module.agent import DQNAgent
from .module.environment import SequenceGuessingConfig, SequenceGuessingEnv


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepTrace:
    """Detailed record for a single inference step."""

    step: int
    action: int
    position: int
    value: int
    reward: float
    done: bool
    action_type: str
    option_name: Optional[str]
    option_status: Optional[str]
    exploration: Optional[bool]
    q_value: Optional[float]
    option_metadata: Dict[str, Any]
    current_guess: Tuple[int, ...]
    feedback: Tuple[int, ...]


@dataclass(frozen=True)
class InferenceOutcome:
    """Container summarising manual inference results."""

    success: bool
    total_reward: float
    total_steps: int
    final_guess: Tuple[int, ...]
    target_sequence: Tuple[int, ...]
    trace: List[StepTrace]
    options_stats: Dict[str, Any]


class StaticSequenceProvider(SequenceGuessingEnv.SequenceProvider):
    """Sequence provider that repeatedly returns a fixed, user supplied sequence."""

    def __init__(self, sequence: Sequence[int]) -> None:
        self.sequence = [int(value) for value in sequence]

    def next_sequence(
        self, expected_length: Optional[int] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        if expected_length is not None and expected_length != len(self.sequence):
            raise ValueError(
                "Target sequence length mismatch: "
                f"expected {expected_length}, got {len(self.sequence)}"
            )
        metadata = {"source": "manual", "description": "User-provided target sequence"}
        return list(self.sequence), metadata


def _to_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(value) for value in values)


def parse_target_sequence(
    cfg: DictConfig,
    env_config: SequenceGuessingConfig,
) -> Tuple[int, ...]:
    """Return validated target sequence from Hydra configuration."""
    if "inference" not in cfg or "target_sequence" not in cfg.inference:
        raise ValueError("Configuration must provide `inference.target_sequence`")

    raw_sequence = cfg.inference.target_sequence
    sequence_list: List[int]

    if isinstance(raw_sequence, str):
        cleaned = raw_sequence.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        if not cleaned:
            raise ValueError("`inference.target_sequence` cannot be empty")
        try:
            sequence_list = [int(part.strip()) for part in cleaned.split(",")]
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "Unable to parse `inference.target_sequence`; supply a list of ints"
            ) from exc
    else:
        sequence_list = [int(value) for value in raw_sequence]

    expected_length = env_config.sequence_length
    if len(sequence_list) != expected_length:
        raise ValueError(
            "Target sequence length mismatch: "
            f"expected {expected_length}, got {len(sequence_list)}"
        )

    if not all(0 <= value < env_config.num_values for value in sequence_list):
        raise ValueError(
            "All target sequence values must be within [0, num_values)"
        )

    return tuple(sequence_list)


def run_manual_inference(
    agent: DQNAgent,
    env: SequenceGuessingEnv,
    *,
    target_sequence: Tuple[int, ...],
    max_steps: Optional[int],
    deterministic: bool,
) -> InferenceOutcome:
    """Run a single inference episode while capturing detailed traces."""
    agent.eval()
    agent.epsilon = 0.0 if deterministic else agent.config.eps_end

    if agent.options_manager:
        agent.options_manager.terminate_active_option()

    steps_limit = max_steps or env.config.max_episode_steps

    with torch.no_grad():
        state = env.reset()

    initial_guess = _to_tuple(env.current_guess.tolist())
    log.info("Initial guess: %s", initial_guess)
    log.info("Target sequence: %s", target_sequence)

    trace: List[StepTrace] = []
    total_reward = 0.0
    step_idx = 0
    done = False

    while not done and step_idx < steps_limit:
        with torch.no_grad():
            action, action_metadata = agent.select_action(state)

        next_state, reward, done, env_info = env.step(int(action))

        action_type = action_metadata.get("action_type", "primitive")
        position = int(env_info["action_decoded"]["position"])
        value = int(env_info["action_decoded"]["value"])
        current_guess = _to_tuple(env_info["current_guess"].tolist())
        feedback_tensor = env_info.get("feedback")
        if isinstance(feedback_tensor, torch.Tensor):
            feedback_values = feedback_tensor.tolist()
        elif feedback_tensor is None:
            feedback_values = []
        else:
            feedback_values = list(feedback_tensor)
        feedback = _to_tuple(feedback_values) if feedback_values else tuple()

        trace.append(
            StepTrace(
                step=step_idx + 1,
                action=int(action),
                position=position,
                value=value,
                reward=float(reward),
                done=bool(done),
                action_type=str(action_type),
                option_name=action_metadata.get("option_name"),
                option_status=action_metadata.get("option_status"),
                exploration=action_metadata.get("exploration"),
                q_value=(
                    float(action_metadata["q_value"])
                    if "q_value" in action_metadata
                    else None
                ),
                option_metadata=dict(action_metadata.get("option_metadata", {}) or {}),
                current_guess=current_guess,
                feedback=feedback,
            )
        )

        log_step(trace[-1])

        total_reward += float(reward)
        state = next_state
        step_idx += 1

        if agent.options_manager and agent.options_manager.get_active_option_name() is None:
            # Ensure epsilon remains deterministic once options terminate
            agent.epsilon = 0.0 if deterministic else agent.config.eps_end

    if agent.options_manager:
        agent.options_manager.terminate_active_option()

    final_guess = _to_tuple(env.current_guess.tolist())
    success = final_guess == target_sequence

    options_stats = (
        agent.options_manager.get_option_stats()
        if agent.options_manager
        else {}
    )

    return InferenceOutcome(
        success=success,
        total_reward=total_reward,
        total_steps=step_idx,
        final_guess=final_guess,
        target_sequence=target_sequence,
        trace=trace,
        options_stats=options_stats,
    )


def log_step(step: StepTrace) -> None:
    """Emit human-readable summary for a single inference step."""
    option_bits: List[str] = []
    if step.option_name:
        option_fragment = f"{step.option_name}/{step.action_type}"
        if step.option_status:
            option_fragment += f"[{step.option_status}]"
        option_bits.append(option_fragment)
    elif step.action_type != "primitive":
        option_bits.append(step.action_type)

    if step.exploration:
        option_bits.append("exploration")
    if step.q_value is not None:
        option_bits.append(f"Q={step.q_value:.2f}")

    option_info = f" | {'; '.join(option_bits)}" if option_bits else ""

    log.info(
        "Step %d: action=%d pos=%d val=%d guess=%s reward=%.2f done=%s%s",
        step.step,
        step.action,
        step.position,
        step.value,
        step.current_guess,
        step.reward,
        step.done,
        option_info,
    )

    if any(step.feedback):
        log.info("        feedback=%s", step.feedback)
    if step.option_metadata:
        log.debug("        option_metadata=%s", step.option_metadata)


@hydra.main(version_base=None, config_path="../conf", config_name="infer")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for manual inference."""
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Running manual inference pipeline")
    log.debug("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = setup_device(cfg.device)
    agent_config = load_agent_config_from_checkpoint(cfg.checkpoint.path)

    env_cfg = cfg.get("env")
    if env_cfg is None:
        raise ValueError("Inference configuration must include an environment profile")

    env_config = instantiate_env_config(env_cfg)
    apply_env_overrides(env_config, cfg.get("env_overrides"))

    target_sequence = parse_target_sequence(cfg, env_config)

    sequence_provider = StaticSequenceProvider(target_sequence)
    env = SequenceGuessingEnv(env_config, sequence_provider=sequence_provider)

    agent_config.max_steps_per_episode = env.config.max_episode_steps
    agent = create_agent(
        cfg.checkpoint.path,
        agent_config,
        env,
        device,
        deterministic=cfg.inference.deterministic,
    )

    outcome = run_manual_inference(
        agent,
        env,
        target_sequence=target_sequence,
        max_steps=cfg.inference.get("max_steps"),
        deterministic=cfg.inference.deterministic,
    )

    log.info(
        "Inference finished in %d steps | final guess=%s | target=%s | total reward=%.2f",
        outcome.total_steps,
        outcome.final_guess,
        outcome.target_sequence,
        outcome.total_reward,
    )

    if outcome.success:
        log.info("Sequence guessed correctly ✅")
    else:
        log.warning("Sequence not guessed correctly ❌")

    if outcome.options_stats:
        log.info("Option usage stats: %s", outcome.options_stats)


if __name__ == "__main__":
    main()
