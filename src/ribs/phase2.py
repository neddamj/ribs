"""Fixed Phase 2 configuration matrix and run-discovery helpers.

The values in this module mirror the implementation specification in the
research plan.  Keeping them in one place prevents the shell scripts, CLI,
validation, and plotting code from silently drifting apart.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import resolved_json
from .data import manifest_hash
from .training import resolve_training_config

PHASE2_FAMILIES = ("dimensional", "vib", "vq", "quantized", "autoencoder")
PHASE2_SPECS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "dimensional": ("dz", (512, 256, 128, 64, 32, 16)),
    "vib": ("beta", (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)),
    "vq": ("codebook_size", (512, 256, 128, 64, 32, 16)),
    "quantized": ("bits", ("FP32", 8, 6, 4, 3, 2)),
    "autoencoder": ("dz", (512, 256, 128, 64, 32)),
}
PHASE2_SEEDS = (0, 1, 2)


def parameter_for_family(family: str) -> str:
    try:
        return PHASE2_SPECS[family.lower()][0]
    except KeyError as exc:
        raise ValueError(f"Unknown Phase 2 family: {family}") from exc


def values_for_family(family: str) -> tuple[Any, ...]:
    try:
        return PHASE2_SPECS[family.lower()][1]
    except KeyError as exc:
        raise ValueError(f"Unknown Phase 2 family: {family}") from exc


def normalize_strength(family: str, value: Any) -> Any:
    """Normalize CLI/YAML values to the types used in resolved configs."""
    family = family.lower()
    if family == "quantized":
        if value is None or str(value).lower() in {"fp32", "none"} or int(value) >= 32:
            return "FP32"
        return int(value)
    if family in {"dimensional", "autoencoder", "vq"}:
        return int(value)
    return float(value)


def strength_from_config(config: dict[str, Any]) -> tuple[str, Any]:
    family = str(config["model"]["family"]).lower()
    parameter = parameter_for_family(family)
    value = config["model"].get(parameter)
    if family == "quantized" and value is None:
        value = "FP32"
    return parameter, normalize_strength(family, value)


def ordinal_strength(family: str, value: Any) -> int:
    """Return the preregistered ordinal scale: 0 weak, max strong."""
    normalized = normalize_strength(family, value)
    values = [normalize_strength(family, candidate) for candidate in values_for_family(family)]
    return values.index(normalized)


def expected_phase2_matrix(
    families: tuple[str, ...] = PHASE2_FAMILIES,
    seeds: tuple[int, ...] = PHASE2_SEEDS,
) -> list[tuple[str, str, Any, int]]:
    return [
        (family, parameter_for_family(family), value, seed)
        for family in families
        for value in values_for_family(family)
        for seed in seeds
    ]


def _comparable(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    value.pop("resume", None)
    if "phase2_decision_record" in value:
        value["phase2_decision_record"].pop("path", None)
    return resolved_json(value)


def completed_run_for_config(output_root: str | Path, config: dict[str, Any]) -> Path | None:
    """Find exactly one completed run with the same resolved configuration."""
    target = _comparable(resolve_training_config(config))
    family = str(config["model"]["family"]).lower()
    matches = []
    for run_dir in (Path(output_root) / family).glob("*"):
        if not (run_dir / "COMPLETED").exists():
            continue
        config_path = run_dir / "resolved_config.yaml"
        if config_path.exists() and _comparable(yaml.safe_load(config_path.read_text())) == target:
            matches.append(run_dir)
    if len(matches) > 1:
        raise ValueError(f"Multiple completed runs match configuration: {matches}")
    return matches[0] if matches else None


def valid_identity_run(output_root: str | Path, seed: int, config: dict[str, Any]) -> Path | None:
    """Find a valid completed identity control without requiring template equality."""
    matches = []
    manifest_path = Path(config["data"]["manifest"])
    expected_manifest_hash = manifest_hash(manifest_path) if manifest_path.exists() else None
    for run_dir in (Path(output_root) / "identity").glob("*"):
        if not (run_dir / "COMPLETED").exists():
            continue
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            continue
        run_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if (
            str(run_config.get("model", {}).get("family", "")).lower() != "identity"
            or int(run_config.get("seed", -1)) != int(seed)
            or int(run_config.get("model", {}).get("dz", -1)) != 512
        ):
            continue
        stored_hash = run_dir / "data_manifest_hash.txt"
        if expected_manifest_hash is not None and (
            not stored_hash.exists() or stored_hash.read_text().strip() != expected_manifest_hash
        ):
            continue
        required = (
            run_dir / "checkpoints" / "best_tune_accuracy.pt",
            run_dir / "history.parquet",
            run_dir / "metrics.json",
        )
        if all(path.exists() for path in required):
            matches.append(run_dir)
    if len(matches) > 1:
        raise ValueError(f"Multiple valid identity runs match seed={seed}: {matches}")
    return matches[0] if matches else None


def phase2_strength_columns(frame):
    """Add explicit ordinal strength metadata for analysis/figures."""
    result = frame.copy()
    if "family" not in result:
        return result
    if "bits" in result:
        result["bits"] = result["bits"].map(
            lambda value: (
                None
                if value is None or (not isinstance(value, str) and bool(pd.isna(value)))
                else str(value)
            )
        )
    parameters, values, ordinals = [], [], []
    for row in result.to_dict("records"):
        family = str(row["family"]).lower()
        parameter, value = strength_from_config({"model": row})
        parameters.append(parameter)
        # A single Parquet column cannot safely mix the FP32 label with
        # numeric controls from the other families. The typed value remains
        # available in dz/beta/codebook_size/bits; this field is a display key.
        values.append(str(value))
        ordinals.append(ordinal_strength(family, value))
    result["strength_parameter"] = parameters
    result["strength_value"] = values
    result["bottleneck_strength_ordinal"] = ordinals
    return result


def decision_record_hash(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_decision_record(path: str | Path) -> dict[str, Any]:
    record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "included_model_families_and_strengths",
        "family_specific_training_changes",
        "primary_phase2_outcomes",
        "final_attack_budgets",
        "seeds",
        "checkpoint_selection_procedure",
        "implementation_deviations",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        missing = sorted(required - set(record or {}))
        raise ValueError(f"Incomplete Phase 2 decision record; missing: {missing}")
    for key in required - {"implementation_deviations"}:
        if not record[key]:
            raise ValueError(f"Phase 2 decision record has unfilled field: {key}")
    seeds = record["seeds"]
    training_seeds = seeds.get("training") if isinstance(seeds, dict) else seeds
    if tuple(int(seed) for seed in training_seeds) != PHASE2_SEEDS:
        raise ValueError("Phase 2 decision record must use training seeds [0, 1, 2]")
    if isinstance(seeds, dict) and (
        int(seeds.get("split", -1)) != 2025 or int(seeds.get("attack", -1)) != 2025
    ):
        raise ValueError("Phase 2 decision record must use split and attack seeds 2025")
    families = record["included_model_families_and_strengths"]
    if not isinstance(families, dict) or set(families) != set(PHASE2_FAMILIES):
        raise ValueError(
            "Phase 2 decision record must contain exactly the registered model families"
        )
    for family, (parameter, expected_values) in PHASE2_SPECS.items():
        specification = families[family]
        if not isinstance(specification, dict) or specification.get("parameter") != parameter:
            raise ValueError(f"Phase 2 {family} must use strength parameter {parameter}")
        observed_values = [
            normalize_strength(family, value) for value in specification.get("values", [])
        ]
        expected_normalized = [normalize_strength(family, value) for value in expected_values]
        if observed_values != expected_normalized:
            raise ValueError(
                f"Phase 2 {family} strengths differ from the registered matrix: "
                f"{observed_values} != {expected_normalized}"
            )
    budgets = record["final_attack_budgets"]
    if not isinstance(budgets, dict):
        raise ValueError("Phase 2 final_attack_budgets must be a mapping")
    input_budget = budgets.get("input_pgd", {})
    latent_budget = budgets.get("latent_pgd", {})
    expected_epsilons = [value / 255 for value in (0, 1, 2, 4, 8, 12, 16)]
    observed_epsilons = [float(value) for value in input_budget.get("epsilons", [])]
    if len(observed_epsilons) != len(expected_epsilons) or any(
        not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12)
        for observed, expected in zip(observed_epsilons, expected_epsilons)
    ):
        raise ValueError("Phase 2 input-PGD epsilon grid differs from the frozen plan")
    if int(input_budget.get("steps", -1)) != 40 or int(input_budget.get("restarts", -1)) != 5:
        raise ValueError("Phase 2 input-PGD must use 40 steps and 5 restarts")
    expected_rhos = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    observed_rhos = [float(value) for value in latent_budget.get("relative_l2_rhos", [])]
    if observed_rhos != expected_rhos:
        raise ValueError("Phase 2 latent-PGD radius grid differs from the frozen plan")
    if (
        int(latent_budget.get("steps", -1)) != 40
        or int(latent_budget.get("restarts", -1)) != 5
        or int(latent_budget.get("vib_eot_samples", -1)) != 32
    ):
        raise ValueError("Phase 2 latent-PGD must use 40 steps, 5 restarts, and VIB EoT=32")
    return record
