"""Run directories and portable result artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from safetensors.torch import load_file, save_file

from .config import run_id
from .utils import environment_info, write_json


class RunDirectory:
    """Write a run atomically and never overwrite a completed run."""

    def __init__(self, config: dict[str, Any], output_root: str | Path | None = None):
        root = Path(output_root or config.get("output_dir", "outputs"))
        family = config.get("model", {}).get("family", "run")
        base = root / family
        base.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while (base / run_id(config, attempt)).exists():
            attempt += 1
        self.final_dir = base / run_id(config, attempt)
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f".{self.final_dir.name}-", dir=base))
        self.config = config

    @property
    def path(self) -> Path:
        return self.temp_dir

    def initialize(self, data_manifest_hash: str | None = None) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        for directory in ("checkpoints", "metrics", "evaluations", "artifacts", "analysis"):
            (self.path / directory).mkdir(parents=True, exist_ok=True)
        (self.path / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=True), encoding="utf-8"
        )
        write_json(self.path / "environment.json", environment_info())
        if data_manifest_hash is not None:
            (self.path / "data_manifest_hash.txt").write_text(
                data_manifest_hash + "\n", encoding="utf-8"
            )

    def complete(self, metrics: dict[str, Any] | None = None) -> Path:
        if metrics is not None:
            write_json(self.path / "metrics.json", metrics)
        (self.path / "COMPLETED").write_text("completed\n", encoding="utf-8")
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run: {self.final_dir}")
        os.replace(self.path, self.final_dir)
        return self.final_dir

    def fail(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pyarrow or fastparquet") from exc
    finally:
        temporary.unlink(missing_ok=True)


def save_latents(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
            str(temporary),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_latents(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))
