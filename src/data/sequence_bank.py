"""Utility for generating reusable target sequence datasets."""

from __future__ import annotations

import json
import math
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
    del params  # uniform sequences ignore per-family parameters for now

    max_attempts = max(50, length * 20)
    for _ in range(max_attempts):
        sequence = [rng.randint(cfg["digit_min"], cfg["digit_max"]) for _ in range(length)]
        if not _matches_special_anchor(sequence, cfg):
            metadata = {
                "distribution": "uniform",
                "base_uniform": sequence.copy(),
            }
            return sequence, metadata

    raise RuntimeError(
        "Unable to sample a uniform sequence that avoids special-family anchors"
    )


def _generate_uniform_base(
    length: int, cfg: Dict[str, Any], rng: random.Random
) -> List[int]:
    return [rng.randint(cfg["digit_min"], cfg["digit_max"]) for _ in range(length)]


def _matches_special_anchor(sequence: List[int], cfg: Dict[str, Any]) -> bool:
    if not sequence:
        return False

    digit_min = int(cfg["digit_min"])
    digit_max = int(cfg["digit_max"])

    def within(value: int) -> bool:
        return digit_min <= value <= digit_max

    first = sequence[0]
    last = sequence[-1]
    length = len(sequence)

    if length > 1 and sequence == sequence[::-1]:
        return True
    if within(0) and first == 0 and last == 0:
        return True
    if within(0) and within(9) and first == 0 and last == 9:
        return True
    if within(1) and length >= 2 and sequence[0] == sequence[1] == 1:
        return True
    if within(5) and first == last == 5:
        return True
    if within(6) and first == last == 6:
        return True
    if within(1) and within(8) and first == 1 and last == 8:
        return True
    if within(8) and within(4) and first == 8 and last == 4:
        return True
    if within(3) and length >= 2 and sequence[0:2] == [3, 1]:
        return True
    if last - first == 4:
        return True

    return False


PI_DIGITS = (
    "3141592653589793238462643383279502884197169399375105820974944592"
    "3078164062862089986280348253421170679"
)


def _ensure_digit_range(value: int, cfg: Dict[str, Any]) -> int:
    return max(cfg["digit_min"], min(cfg["digit_max"], value))


@register_family("palindrome")
def generate_palindrome(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if length <= 0:
        raise ValueError("Sequence length must be positive")

    if not (cfg["digit_min"] <= 0 <= cfg["digit_max"]):
        raise ValueError("Digit range must include 0 for palindrome starting condition")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    sequence = base_uniform.copy()

    first_digit = _ensure_digit_range(params.get("first_digit", 0), cfg)
    requested_last = params.get("last_digit")
    if requested_last is None:
        last_digit = first_digit
    else:
        last_digit = _ensure_digit_range(requested_last, cfg)

    if first_digit != last_digit:
        raise ValueError("First and last digits must match for a palindrome sequence")

    digit_choices = params.get("digit_choices")
    if digit_choices is not None:
        normalized_choices = []
        for value in digit_choices:
            try:
                digit = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid digit choice '{value}'") from exc
            if cfg["digit_min"] <= digit <= cfg["digit_max"]:
                normalized_choices.append(_ensure_digit_range(digit, cfg))
        allowed_digits = sorted(set(normalized_choices))
        if not allowed_digits:
            raise ValueError("No valid digit choices available within digit range")
    else:
        allowed_digits = list(range(cfg["digit_min"], cfg["digit_max"] + 1))

    if first_digit not in allowed_digits:
        allowed_digits = sorted(allowed_digits + [first_digit])

    if length >= 1:
        sequence[0] = first_digit
    if length >= 2:
        sequence[-1] = first_digit

    for idx in range(length // 2):
        if idx == 0:
            value = first_digit
        else:
            value = rng.choice(allowed_digits)
        sequence[idx] = value
        sequence[length - idx - 1] = value

    if length % 2 == 1:
        mid = length // 2
        middle_digit = params.get("middle_digit")
        if middle_digit is not None:
            value = _ensure_digit_range(int(middle_digit), cfg)
        else:
            value = first_digit if length == 1 else rng.choice(allowed_digits)
        sequence[mid] = value

    metadata = {
        "transformation": "palindrome_symmetric",
        "starting_condition": {"first_digit": first_digit, "last_digit": last_digit},
        "digit_choices": allowed_digits,
        "base_uniform": base_uniform,
    }
    return sequence, metadata


@register_family("same_number_block")
def generate_same_number_block(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if length <= 0:
        raise ValueError("Sequence length must be positive")

    if not (cfg["digit_min"] <= 1 <= cfg["digit_max"]):
        raise ValueError(
            "Digit range must include 1 for same number block starting condition"
        )

    base_uniform = _generate_uniform_base(length, cfg, rng)
    max_block = int(params.get("max_block_size", max(2, min(4, length))))
    max_block = max(2, min(max_block, length))

    sequence: List[int] = []
    block_lengths: List[int] = []

    first_block_len = min(length, max(2, rng.randint(2, max_block)))
    sequence.extend([1] * first_block_len)
    block_lengths.append(first_block_len)

    previous_digit = 1
    idx = first_block_len
    while idx < length:
        digit = rng.randint(cfg["digit_min"], cfg["digit_max"])
        if digit == previous_digit:
            digit = (digit + 1 - cfg["digit_min"]) % (cfg["digit_max"] - cfg["digit_min"] + 1)
            digit += cfg["digit_min"]

        remaining = length - idx
        block_len = rng.randint(1, min(remaining, max_block))
        sequence.extend([digit] * block_len)
        block_lengths.append(block_len)
        previous_digit = digit
        idx += block_len

    sequence = sequence[:length]
    if length >= 1:
        sequence[0] = 1
    if length >= 2:
        sequence[1] = 1

    total_blocks = sum(block_lengths)
    if total_blocks > length:
        overflow = total_blocks - length
        block_lengths[-1] = max(1, block_lengths[-1] - overflow)

    final_blocks: List[int] = []
    if sequence:
        current_digit = sequence[0]
        run_length = 1
        for value in sequence[1:]:
            if value == current_digit:
                run_length += 1
            else:
                final_blocks.append(run_length)
                current_digit = value
                run_length = 1
        final_blocks.append(run_length)

    metadata = {
        "transformation": "same_number_block",
        "starting_condition": {"first_digit": 1, "second_digit": 1},
        "block_lengths": final_blocks,
        "base_uniform": base_uniform,
    }
    return sequence, metadata


@register_family("gaussian_sequence")
def generate_gaussian_sequence(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    mean = float(params.get("mean", 5.0))
    variance = float(params.get("variance", 2.0))
    std = math.sqrt(max(variance, 0.0))

    anchor_value = int(params.get("anchor_value", 5))
    if not (cfg["digit_min"] <= anchor_value <= cfg["digit_max"]):
        raise ValueError("Digit range must include anchor value for gaussian sequence")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    samples = np_rng.normal(loc=mean, scale=std if std > 0 else 1.0, size=length)
    digits = np.rint(samples).astype(int)
    clipped = np.clip(digits, cfg["digit_min"], cfg["digit_max"]).tolist()

    if length >= 1:
        clipped[0] = _ensure_digit_range(anchor_value, cfg)
    if length >= 2:
        clipped[-1] = _ensure_digit_range(anchor_value, cfg)

    metadata = {
        "transformation": "gaussian_sequence",
        "mean": mean,
        "variance": variance,
        "anchor_value": anchor_value,
        "starting_condition": {"first_digit": anchor_value, "last_digit": anchor_value},
        "base_uniform": base_uniform,
    }
    return clipped, metadata


@register_family("monotonic_sequence")
def generate_monotonic_sequence(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if length <= 0:
        raise ValueError("Sequence length must be positive")

    if not (cfg["digit_min"] <= 6 <= cfg["digit_max"]):
        raise ValueError("Digit range must include 6 for sum targeter starting condition")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    sequence = [0] * length
    start_digit = _ensure_digit_range(params.get("start_digit", 0), cfg)
    end_digit = _ensure_digit_range(params.get("end_digit", 9), cfg)
    if start_digit > end_digit:
        start_digit, end_digit = end_digit, start_digit

    if length >= 1:
        sequence[0] = start_digit

    current = start_digit if length >= 1 else 0
    for idx in range(1, length - 1):
        positions_left = length - idx - 1
        min_val = current
        max_val = end_digit - positions_left
        max_val = max(min_val, min(max_val, cfg["digit_max"]))
        next_value = rng.randint(min_val, max_val)
        sequence[idx] = next_value
        current = next_value

    if length >= 2:
        sequence[-1] = end_digit
        if sequence[-2] > sequence[-1]:
            sequence[-2] = sequence[-1]

    metadata = {
        "transformation": "monotonic_increasing",
        "starting_condition": {
            "first_digit": start_digit,
            "last_digit": end_digit,
        },
        "base_uniform": base_uniform,
    }
    return sequence, metadata


@register_family("sum_targeter")
def generate_sum_targeter(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if length <= 0:
        raise ValueError("Sequence length must be positive")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    target = int(params.get("target_sum", 60))
    min_sum = cfg["digit_min"] * length
    max_sum = cfg["digit_max"] * length
    target = max(min_sum, min(max_sum, target))

    sequence = [cfg["digit_min"]] * length
    if length >= 1:
        sequence[0] = _ensure_digit_range(6, cfg)
    if length >= 2:
        sequence[-1] = _ensure_digit_range(6, cfg)

    indices = list(range(1, length - 1))
    if not indices and target != sum(sequence):
        raise ValueError("Target sum incompatible with sequence length for Sum Targeter")

    current_sum = sum(sequence)
    difference = target - current_sum

    attempts = 0
    limit = length * cfg.get("digit_max", 9) * 10
    while difference != 0 and attempts < limit:
        attempts += 1
        if not indices:
            break
        idx = rng.choice(indices)
        if difference > 0:
            if sequence[idx] >= cfg["digit_max"]:
                continue
            sequence[idx] += 1
            difference -= 1
        else:
            if sequence[idx] <= cfg["digit_min"]:
                continue
            sequence[idx] -= 1
            difference += 1

    if difference != 0:
        raise ValueError("Unable to satisfy target sum within constraints")

    metadata = {
        "transformation": "sum_targeter",
        "target_sum": target,
        "starting_condition": {"first_digit": 6, "last_digit": 6},
        "base_uniform": base_uniform,
    }
    return sequence, metadata


@register_family("high_low_alternator")
def generate_high_low_alternator(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    low_digits = params.get("low_digits", [0, 1, 2])
    high_digits = params.get("high_digits", [7, 8, 9])

    low_choices = [d for d in low_digits if cfg["digit_min"] <= d <= cfg["digit_max"]]
    high_choices = [d for d in high_digits if cfg["digit_min"] <= d <= cfg["digit_max"]]
    if not low_choices or not high_choices:
        raise ValueError("Digit range incompatible with high/low alternator choices")
    if 1 not in low_choices:
        raise ValueError("Low digit options must include 1 for starting condition")
    if 8 not in high_choices:
        raise ValueError("High digit options must include 8 for starting condition")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    sequence = []
    for idx in range(length):
        candidates = low_choices if idx % 2 == 0 else high_choices
        value = rng.choice(candidates)
        sequence.append(value)

    if length >= 1:
        sequence[0] = 1
    if length >= 2:
        sequence[-1] = 8

    metadata = {
        "transformation": "high_low_alternator",
        "starting_condition": {"first_digit": 1, "last_digit": 8},
        "base_uniform": base_uniform,
    }
    return sequence, metadata


@register_family("repetitive_pattern")
def generate_repetitive_pattern(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    __: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    pattern = params.get("pattern", [8, 4])
    if not pattern:
        raise ValueError("Pattern must contain at least one digit")
    for value in pattern:
        if not (cfg["digit_min"] <= int(value) <= cfg["digit_max"]):
            raise ValueError("Pattern digits must fall within configured digit range")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    sequence = [pattern[idx % len(pattern)] for idx in range(length)]
    if length >= 1:
        sequence[0] = pattern[0]
    if length >= 2:
        sequence[-1] = pattern[1 % len(pattern)]

    metadata = {
        "transformation": "repetitive_pattern",
        "pattern": pattern,
        "starting_condition": {"first_digit": 8, "last_digit": 4},
        "base_uniform": base_uniform,
    }
    return sequence, metadata


@register_family("digits_of_pi")
def generate_digits_of_pi(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if length > len(PI_DIGITS):
        raise ValueError("Requested length exceeds available precomputed pi digits")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    digits = [int(char) for char in PI_DIGITS[:length]]

    metadata = {
        "transformation": "digits_of_pi",
        "starting_condition": {"prefix": [3, 1]},
        "base_uniform": base_uniform,
    }
    return digits, metadata


@register_family("difference_maintaining")
def generate_difference_maintaining(
    length: int,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    _: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if length < 2:
        raise ValueError("Sequence length must be at least 2 for difference maintaining")

    base_uniform = _generate_uniform_base(length, cfg, rng)
    range_width = int(params.get("range_width", 4))
    if range_width <= 0:
        raise ValueError("range_width must be positive")
    max_start = cfg["digit_max"] - range_width
    min_start = cfg["digit_min"]
    if min_start > max_start:
        raise ValueError("Digit range too small for difference maintaining sequence")

    start_value = rng.randint(min_start, max_start)
    end_value = start_value + range_width

    sequence = [start_value]
    midpoint = start_value + range_width // 2
    for _ in range(length - 2):
        sequence.append(midpoint)
    if length >= 2:
        sequence.append(end_value)

    metadata = {
        "transformation": "difference_maintaining",
        "range_width": range_width,
        "starting_condition": {"first_last_difference": range_width},
        "base_uniform": base_uniform,
    }
    return sequence, metadata


def _prepare_cfg(cfg: DictConfig) -> Dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Configuration must resolve to a dictionary")
    return container


def generate_bank(cfg: DictConfig) -> Tuple[List[Dict[str, Any]], SequenceBankSummary]:
    """Generate a bank of unique sequences for a single family type.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (records, summary)
    """
    resolved_cfg = _prepare_cfg(cfg)
    rng = random.Random(resolved_cfg["seed"])
    np_rng = np.random.default_rng(resolved_cfg["seed"])

    sequence_length = int(resolved_cfg.get("sequence_length", 0))
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")

    count = int(resolved_cfg.get("count", 0))
    if count <= 0:
        raise ValueError("count must be a positive integer")

    # Get family configuration
    family_cfg = getattr(cfg, "family", None)
    if family_cfg is None:
        raise KeyError("Configuration must contain a 'family' section")

    family_container = OmegaConf.to_container(family_cfg, resolve=True)
    if not isinstance(family_container, dict):
        raise TypeError("Family config must be a mapping")

    family_name = str(family_container.get("name", "")).strip()
    if not family_name:
        raise ValueError("family.name must be provided")

    # Get the generator for this family
    generator = GENERATOR_REGISTRY.get(family_name)
    if generator is None:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise KeyError(
            f"Unknown family '{family_name}'. Available families: {available}"
        )

    entry_params = family_container.get("params", {}) or {}

    # Generate unique sequences
    records: List[Dict[str, Any]] = []
    seen_sequences: set = set()

    idx = 0
    attempts = 0
    max_attempts = count * 10  # Allow more attempts to reach target

    while len(records) < count and attempts < max_attempts:
        attempts += 1
        try:
            sequence, metadata = generator(
                sequence_length, entry_params, resolved_cfg, rng, np_rng
            )
            validate_sequence(sequence, resolved_cfg)

            # Check for duplicates
            sequence_tuple = tuple(sequence)
            if sequence_tuple in seen_sequences:
                continue

            seen_sequences.add(sequence_tuple)
            metadata = dict(metadata or {})
            metadata.setdefault("family_params", dict(entry_params))

            record = {
                "sequence": sequence,
                "family": family_name,
                "sequence_length": len(sequence),
                "metadata": metadata,
                "index": idx,
                "seed": resolved_cfg["seed"],
            }
            records.append(record)
            idx += 1
        except (ValueError, RuntimeError) as e:
            if attempts % 1000 == 0:
                print(f"Warning: Failed attempt {attempts} for {family_name}: {e}")
            continue

    if not records:
        raise ValueError(f"No sequences generated for family '{family_name}'")

    if len(records) < count:
        print(f"Warning: Only generated {len(records)}/{count} unique sequences for '{family_name}' after {attempts} attempts")

    # Create summary
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

    return records, summary


def _format_records(records: List[Dict[str, Any]], include_id: bool = True) -> List[Dict[str, Any]]:
    formatted = []
    for item in records:
        record_dict = {
            "sequence": list(map(int, item["sequence"])),
            "family": item["family"],
            "sequence_length": int(item["sequence_length"]),
            "index": int(item["index"]),
            "seed": int(item["seed"]),
            "metadata": item.get("metadata", {}),
        }
        if include_id and "id" in item:
            record_dict["id"] = item["id"]
        formatted.append(record_dict)
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
    prepared = _format_records(records, include_id=False)
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

    # Generate sequences for the specified family
    records, summary = generate_bank(data_cfg)
    output_path = write_outputs(records, summary, data_cfg)
    print_summary(output_path, summary)


if __name__ == "__main__":
    main()
