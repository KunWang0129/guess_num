"""Utility for generating reusable target sequence datasets."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


SequenceGenerator = Callable[
    [int, Dict[str, Any], Dict[str, Any], random.Random, np.random.Generator],
    Tuple[List[int], Dict[str, Any]],
]

GENERATOR_REGISTRY: Dict[str, SequenceGenerator] = {}


@dataclass
class SequenceBankSummary:
    """Lightweight summary returned after dataset generation."""

    total_sequences: int
    family_counts: Dict[str, int]
    length_min: int
    length_max: int
    length_mean: float
    length_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sequences": self.total_sequences,
            "family_counts": dict(self.family_counts),
            "length_min": self.length_min,
            "length_max": self.length_max,
            "length_mean": round(self.length_mean, 3),
            "length_std": round(self.length_std, 3),
        }


def register_family(name: str) -> Callable[[SequenceGenerator], SequenceGenerator]:
    def decorator(func: SequenceGenerator) -> SequenceGenerator:
        GENERATOR_REGISTRY[name] = func
        return func

    return decorator


def validate_sequence(sequence: List[int], cfg: Dict[str, Any]) -> None:
    low = int(cfg["digit_min"])
    high = int(cfg["digit_max"])
    if not sequence:
        raise ValueError("Generated sequence is empty")
    for value in sequence:
        if not low <= int(value) <= high:
            raise ValueError(
                f"Value {value} outside allowed range [{low}, {high}]"
            )


@register_family("uniform")
def generate_uniform(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    sequence = [rng.randint(cfg["digit_min"], cfg["digit_max"]) for _ in range(length)]
    return sequence, {}


@register_family("complement_pairs")
def generate_complement_pairs(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    target_sum = int(params.get("target_sum", cfg.get("target_sum", 10)))
    low = int(cfg["digit_min"])
    high = int(cfg["digit_max"])
    lower_bound = max(low, target_sum - high)
    upper_bound = min(high, target_sum - low)
    if lower_bound > upper_bound:
        raise ValueError(
            "Complement pair generation impossible with current digit range and target"
        )

    sequence = [0] * length
    for idx in range(length // 2):
        left = rng.randint(lower_bound, upper_bound)
        right = target_sum - left
        sequence[idx] = left
        sequence[length - idx - 1] = right

    if length % 2 == 1:
        sequence[length // 2] = rng.randint(low, high)

    return sequence, {"target_sum": target_sum}


@register_family("parity_lock")
def generate_parity_lock(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    parity = params.get("parity", "odd")
    parity = parity.lower()
    if parity not in {"odd", "even"}:
        raise ValueError("parity must be 'odd' or 'even'")

    remainder = 1 if parity == "odd" else 0
    allowed = [
        value
        for value in range(cfg["digit_min"], cfg["digit_max"] + 1)
        if value % 2 == remainder
    ]
    if not allowed:
        raise ValueError("No digits available for selected parity and digit range")

    sequence = [rng.choice(allowed) for _ in range(length)]
    return sequence, {"parity": parity, "allowed_digits": allowed}


@register_family("gaussian_centered")
def generate_gaussian_centered(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    mean = float(params.get("mean", 5.0))
    std = float(params.get("std", 1.5))
    digits = np_rng.normal(loc=mean, scale=std, size=length)
    rounded = np.rint(digits).astype(int)
    clipped = np.clip(rounded, cfg["digit_min"], cfg["digit_max"]).tolist()
    return clipped, {"mean": mean, "std": std}


@register_family("progression")
def generate_progression(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    start_range = params.get("start_range", [cfg["digit_min"], cfg["digit_max"]])
    if len(start_range) != 2:
        raise ValueError("start_range must contain [min, max]")
    start_low = int(start_range[0])
    start_high = int(start_range[1])
    if start_low > start_high:
        start_low, start_high = start_high, start_low

    step_options = params.get("step_options", [-2, -1, 1, 2])
    if not step_options:
        raise ValueError("step_options must provide at least one step value")

    noise_prob = float(params.get("noise_prob", 0.0))
    noise_magnitude = int(params.get("noise_magnitude", 1))

    start = rng.randint(start_low, start_high)
    step = rng.choice(step_options)

    sequence: List[int] = []
    noisy_indices: List[int] = []
    for idx in range(length):
        value = start + idx * step
        if noise_prob > 0 and rng.random() < noise_prob:
            jitter = rng.randint(-noise_magnitude, noise_magnitude)
            value += jitter
            noisy_indices.append(idx)
        value = max(cfg["digit_min"], min(cfg["digit_max"], value))
        sequence.append(value)

    metadata = {
        "start": start,
        "step": step,
        "noise_prob": noise_prob,
        "noise_magnitude": noise_magnitude,
        "noisy_indices": noisy_indices,
    }
    return sequence, metadata


def _prepare_cfg(cfg: DictConfig) -> Dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Configuration must resolve to a dictionary")
    return container


def generate_bank(cfg: DictConfig) -> Tuple[List[Dict[str, Any]], SequenceBankSummary]:
    resolved_cfg = _prepare_cfg(cfg)
    rng = random.Random(resolved_cfg["seed"])
    np_rng = np.random.default_rng(resolved_cfg["seed"])

    family_cfg = getattr(cfg, "family", None)
    if family_cfg is None:
        raise KeyError("Configuration must contain a 'family' section")

    family_container = OmegaConf.to_container(family_cfg, resolve=True)
    if not isinstance(family_container, dict):
        raise TypeError("Family config must be a mapping")

    family_name = str(family_container.get("name", "")).strip()
    if not family_name:
        raise ValueError("family.name must be provided")

    count = int(resolved_cfg.get("count", 0))
    if count <= 0:
        raise ValueError("count must be a positive integer")

    params = family_container.get("params", {}) or {}

    sequence_length = int(resolved_cfg.get("sequence_length", 0))
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")

    generator = GENERATOR_REGISTRY.get(family_name)
    if generator is None:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise KeyError(
            f"Unknown family '{family_name}'. Available families: {available}"
        )

    records: List[Dict[str, Any]] = []
    for sample_index in range(count):
        sequence, metadata = generator(
            sequence_length, params, resolved_cfg, rng, np_rng
        )
        validate_sequence(sequence, resolved_cfg)
        metadata = dict(metadata or {})
        record = {
            "id": f"{family_name}_{sample_index}",
            "sequence": sequence,
            "family": family_name,
            "sequence_length": len(sequence),
            "metadata": metadata,
        }
        record["metadata"].update({"family_params": dict(params)})
        records.append(record)

    if not records:
        raise ValueError("No sequences generated; check family counts in configuration")

    lengths = [entry["sequence_length"] for entry in records]
    family_counts = Counter(entry["family"] for entry in records)

    summary = SequenceBankSummary(
        total_sequences=len(records),
        family_counts=dict(family_counts),
        length_min=min(lengths),
        length_max=max(lengths),
        length_mean=float(np.mean(lengths)),
        length_std=float(np.std(lengths)),
    )

    for idx, record in enumerate(records):
        record["index"] = idx
        record["seed"] = resolved_cfg["seed"]

    return records, summary


def _format_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for item in records:
        formatted.append(
            {
                "id": item["id"],
                "sequence": list(map(int, item["sequence"])),
                "family": item["family"],
                "sequence_length": int(item["sequence_length"]),
                "index": int(item["index"]),
                "seed": int(item["seed"]),
                "metadata": item.get("metadata", {}),
            }
        )
    return formatted


def write_outputs(
    records: List[Dict[str, Any]],
    summary: SequenceBankSummary,
    cfg: DictConfig,
) -> Path:
    resolved_cfg = _prepare_cfg(cfg)
    base_dir = Path(get_original_cwd()) / resolved_cfg["output"]["dir"]
    base_dir.mkdir(parents=True, exist_ok=True)

    include_ts = resolved_cfg["output"].get("include_timestamp", True)
    prefix_base = resolved_cfg["output"].get("prefix", "sequence_bank")
    family_name = ""
    family_cfg = resolved_cfg.get("family")
    if isinstance(family_cfg, dict):
        family_name = str(family_cfg.get("name", "")).strip()
    prefix = prefix_base
    if family_name and family_name not in prefix_base:
        prefix = f"{prefix_base}_{family_name}"

    # Append "_test" to prefix if test flag is true
    if resolved_cfg.get("test", False):
        prefix = f"{prefix}_test"

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") if include_ts else ""
    stem = f"{prefix}_{run_id}" if run_id else prefix

    data_format = resolved_cfg["output"].get("format", "jsonl").lower()
    if data_format != "jsonl":
        raise ValueError("Currently only 'jsonl' output format is supported")

    data_path = base_dir / f"{stem}.jsonl"
    prepared = _format_records(records)
    with data_path.open("w", encoding="utf-8") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    if resolved_cfg.get("manifest", True):
        manifest_path = base_dir / f"{stem}_manifest.json"
        manifest_payload = {
            "summary": summary.to_dict(),
            "config": _prepare_cfg(cfg),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest_payload, manifest_file, ensure_ascii=False, indent=2)

    return data_path


def print_summary(path: Path, summary: SequenceBankSummary) -> None:
    print("Sequence bank generated:")
    print(f"  file: {path}")
    print(f"  total sequences: {summary.total_sequences}")
    print(f"  length stats: min={summary.length_min}, max={summary.length_max}, "
          f"mean={summary.length_mean:.2f}, std={summary.length_std:.2f}")
    print("  families:")
    for family, count in sorted(summary.family_counts.items()):
        print(f"    - {family}: {count} sequences")


@hydra.main(version_base=None, config_path="../../conf", config_name="data/sequence_bank")
def main(cfg: DictConfig) -> None:
    # Extract the data config from the nested structure
    data_cfg = cfg.data if 'data' in cfg else cfg

    records, summary = generate_bank(data_cfg)
    output_path = write_outputs(records, summary, data_cfg)
    print_summary(output_path, summary)


if __name__ == "__main__":
    main()
