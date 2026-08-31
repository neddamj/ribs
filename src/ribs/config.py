"""Small, explicit configuration helpers used by every command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def set_override(config: dict[str, Any], key: str, value: str) -> None:
    """Set a dotted YAML key, parsing scalars/lists with YAML semantics."""
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise TypeError(f"Cannot descend into non-mapping config key: {part}")
    target[parts[-1]] = yaml.safe_load(value)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have KEY=VALUE form: {override}")
        key, value = override.split("=", 1)
        set_override(result, key, value)
    return result


def resolved_json(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, indent=2, default=str) + "\n"


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(resolved_json(config).encode("utf-8")).hexdigest()[:12]


def run_id(config: dict[str, Any], attempt: int = 0) -> str:
    family = config.get("model", {}).get("family", "run")
    seed = config.get("seed", 0)
    return f"{family}-seed{seed}-{config_hash(config)}-attempt{attempt}"


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "train", "attack"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    for key in ("root", "manifest"):
        if key not in config["data"]:
            raise ValueError(f"Missing data.{key}")
    for key in ("batch_size", "effective_batch_size", "epochs", "learning_rate"):
        if key not in config["train"]:
            raise ValueError(f"Missing train.{key}")
    family = str(config["model"].get("family", "dimensional")).lower()
    if (
        family in {"dimensional", "vib", "quantized", "quantized_continuous", "autoencoder"}
        and "dz" not in config["model"]
    ):
        raise ValueError(f"Missing model.dz for family={family}")
    if family == "vib" and "beta" not in config["model"]:
        raise ValueError("Missing model.beta for VIB")
    if family == "vq" and "codebook_size" not in config["model"]:
        raise ValueError("Missing model.codebook_size for VQ")
