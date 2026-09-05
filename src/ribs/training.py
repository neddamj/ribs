"""Training and clean evaluation loops."""

from __future__ import annotations

import copy
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .artifacts import RunDirectory, save_frame
from .config import validate_config
from .data import ImagenetteDataset, manifest_hash
from .models import AutoencoderBottleneck, BottleneckModel, create_model
from .utils import make_logger, seed_everything, write_json


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data = config["data"]
    dataset = ImagenetteDataset(
        data["root"],
        data["manifest"],
        split,
        int(data.get("image_size", 224)),
        data.get("size", "320px"),
    )
    seeds = config.get("random_seeds", {})
    generator_seed = int(
        seeds.get("data_order", int(config.get("seed", 0)) + 17)
        if shuffle
        else seeds.get("evaluation_order", int(config.get("seed", 0)) + 31)
    )
    generator = torch.Generator().manual_seed(generator_seed)
    worker_base = int(
        seeds.get("augmentation", int(config.get("seed", 0)) + 1000)
        if shuffle
        else seeds.get("evaluation_workers", int(config.get("seed", 0)) + 2000)
    )

    def worker_init(worker_id: int) -> None:
        seed_everything(worker_base + worker_id, deterministic=False)

    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_init,
    )


def _training_batches(loader: DataLoader, augmentation_generator: torch.Generator):
    """Keep main-process augmentation draws separate from model randomness."""
    if loader.num_workers > 0:
        yield from loader
        return
    iterator = iter(loader)
    while True:
        model_rng_state = torch.get_rng_state()
        torch.set_rng_state(augmentation_generator.get_state())
        try:
            batch = next(iterator)
        except StopIteration:
            augmentation_generator.set_state(torch.get_rng_state())
            torch.set_rng_state(model_rng_state)
            break
        augmentation_generator.set_state(torch.get_rng_state())
        torch.set_rng_state(model_rng_state)
        yield batch


def _optimizer(model: BottleneckModel, config: dict[str, Any]) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    train_cfg = config["train"]
    return AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def _scheduler(optimizer: AdamW, config: dict[str, Any], steps_per_epoch: int) -> LambdaLR:
    train_cfg = config["train"]
    total = int(train_cfg["epochs"]) * max(1, steps_per_epoch)
    warmup = int(train_cfg.get("warmup_epochs", 5)) * max(1, steps_per_epoch)
    minimum = float(train_cfg.get("minimum_learning_rate", 1e-6))
    initial = float(train_cfg["learning_rate"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1e-8, (step + 1) / max(1, warmup))
        fraction = min(1.0, (step - warmup) / max(1, total - warmup))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(fraction * 3.141592653589793))).item()
        return (minimum + (initial - minimum) * cosine) / initial

    return LambdaLR(optimizer, schedule)


def _autocast(device: torch.device, config: dict[str, Any]):
    mode = str(config["train"].get("mixed_precision", "auto")).lower()
    enabled = device.type == "cuda" and mode not in {"false", "none", "off"}
    dtype = (
        torch.bfloat16
        if mode in {"bf16", "auto"} and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return (
        torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
        if enabled
        else nullcontext()
    )


def batch_loss(
    model: BottleneckModel, batch: dict[str, Any], config: dict[str, Any], device: torch.device
) -> tuple[Tensor, dict[str, Any]]:
    images = batch["image"].to(device, non_blocking=True)
    labels = torch.as_tensor(batch["label"], device=device)
    output = model(images, sample=True)
    if isinstance(model, AutoencoderBottleneck):
        reconstruction = output.metadata["reconstruction"]
        loss = F.mse_loss(reconstruction, images)
        return loss, {"loss": float(loss.detach()), "mse": float(loss.detach())}
    ce = F.cross_entropy(
        output.logits, labels, label_smoothing=float(config["train"].get("label_smoothing", 0.0))
    )
    loss = ce
    values = {"loss": float(ce.detach()), "ce": float(ce.detach())}
    for name, aux in output.aux_losses.items():
        coefficient = model.beta if name == "kl" and hasattr(model, "beta") else 1.0
        loss = loss + coefficient * aux
        values[name] = float(aux.detach())
    if "code_indices" in output.metadata:
        codes = output.metadata["code_indices"].detach().flatten()
        counts = torch.bincount(codes, minlength=int(getattr(model, "codebook_size", 1))).float()
        values["codebook_counts"] = counts.cpu().tolist()
    values["loss"] = float(loss.detach())
    return loss, values


def resolve_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize all seed defaults before hashing or writing a run config."""
    config = copy.deepcopy(config)
    validate_config(config)
    seed = int(config.get("seed", 0))
    seeds = config.setdefault("random_seeds", {})
    seed_defaults = {
        "model_initialization": seed,
        "data_order": seed + 17,
        "augmentation": seed + 1000,
        "stochastic_bottleneck": seed + 3000,
        "evaluation_order": seed + 31,
        "evaluation_workers": seed + 2000,
        "attack": int(config.get("attack", {}).get("seed", 2025)),
    }
    for name, value in seed_defaults.items():
        seeds.setdefault(name, value)
    config["attack"].setdefault("seed", config["random_seeds"]["attack"])
    return config


@torch.no_grad()
def evaluate_reconstruction_mse(
    model: AutoencoderBottleneck, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    squared_error = 0.0
    element_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        reconstruction = model(images, sample=False).metadata["reconstruction"]
        squared_error += float((reconstruction - images).square().sum())
        element_count += images.numel()
    return squared_error / max(1, element_count)


@torch.no_grad()
def evaluate_clean(
    model: BottleneckModel, loader: DataLoader, device: torch.device, sample_count: int = 1
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = torch.as_tensor(batch["label"], device=device)
        if sample_count <= 1:
            output = model(images, sample=False)
            logits = output.logits
        else:
            logits_sum = 0.0
            output = model(images, sample=True)
            logits_sum = torch.softmax(output.logits, dim=-1)
            for _ in range(sample_count - 1):
                logits_sum = logits_sum + torch.softmax(model(images, sample=True).logits, dim=-1)
            logits = logits_sum / sample_count
        predictions = logits.argmax(dim=-1)
        for sample_id, label, prediction in zip(
            batch["sample_id"], labels.cpu(), predictions.cpu()
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(label),
                    "clean_prediction": int(prediction),
                    "clean_correct": bool(label == prediction),
                }
            )
    frame = pd.DataFrame(rows)
    return {"clean_accuracy": float(frame.clean_correct.mean())}, frame


def train_model(config: dict[str, Any]) -> Path:
    config = resolve_training_config(config)
    seed = int(config.get("seed", 0))
    seed_everything(int(config["random_seeds"]["model_initialization"]))
    run = RunDirectory(config)
    data_hash = manifest_hash(config["data"]["manifest"])
    run.initialize(data_hash)
    logger = make_logger(run.path / "logs.txt")
    logger.info("Starting training with seed=%s", seed)
    device = choose_device(config)
    model = create_model(config).to(device)
    train_loader = _loader(config, "development_train", True)
    tune_loader = _loader(config, "development_tune", False)
    optimizer = _optimizer(model, config)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
        and str(config["train"].get("mixed_precision", "auto")).lower()
        not in {"false", "none", "off"},
    )
    accumulation = max(
        1,
        int(config["train"].get("effective_batch_size", 128))
        // max(1, int(config["train"].get("batch_size", 128))),
    )
    physical_batch_size = int(config["train"].get("batch_size", 128))
    effective_batch_size = int(config["train"].get("effective_batch_size", 128))
    if effective_batch_size % physical_batch_size != 0:
        raise ValueError("effective_batch_size must be an integer multiple of batch_size")
    optimizer_steps_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    scheduler = _scheduler(optimizer, config, optimizer_steps_per_epoch)
    history: list[dict[str, Any]] = []
    is_autoencoder = isinstance(model, AutoencoderBottleneck)
    best_metric = float("inf") if is_autoencoder else -1.0
    best_epoch = -1
    collapse_epochs = 0
    codebook_collapsed = False
    augmentation_generator = torch.Generator().manual_seed(
        int(config["random_seeds"]["augmentation"])
    )
    start_epoch = 0
    resume_path = config.get("resume")
    if resume_path:
        resume_path = Path(resume_path)
        state = torch.load(resume_path, map_location=device, weights_only=False)
        previous_config = state.get("config", {})
        current_comparable = {key: value for key, value in config.items() if key != "resume"}
        previous_comparable = {
            key: value for key, value in previous_config.items() if key != "resume"
        }
        if previous_comparable != current_comparable:
            raise ValueError("Resume checkpoint configuration does not exactly match this run")
        previous_hash = manifest_hash(previous_config["data"]["manifest"])
        if previous_hash != data_hash:
            raise ValueError("Resume checkpoint uses different manifest contents")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        if "torch_rng_state" in state:
            torch_rng_state = state["torch_rng_state"]
            # ``map_location=device`` moves every tensor in the checkpoint to
            # CUDA, but the default torch generator is CPU-backed.
            torch.set_rng_state(torch_rng_state.cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng_state"]])
        if "data_loader_rng_state" in state and train_loader.generator is not None:
            train_loader.generator.set_state(state["data_loader_rng_state"].cpu())
        if "augmentation_rng_state" in state:
            augmentation_generator.set_state(state["augmentation_rng_state"].cpu())
        start_epoch = int(state["epoch"]) + 1
        previous_run = resume_path.parent.parent
        history_path = previous_run / "history.parquet"
        if history_path.exists():
            history = pd.read_parquet(history_path).to_dict("records")
        tune_metrics_path = (
            previous_run
            / "metrics"
            / ("tune_reconstruction.json" if is_autoencoder else "tune_clean.json")
        )
        if tune_metrics_path.exists():
            previous_metrics = json.loads(tune_metrics_path.read_text())
            best_metric = float(
                previous_metrics["mse"] if is_autoencoder else previous_metrics["clean_accuracy"]
            )
            best_epoch = int(previous_metrics["epoch"])
        previous_best = (
            previous_run
            / "checkpoints"
            / ("best_tune_mse.pt" if is_autoencoder else "best_tune_accuracy.pt")
        )
        if previous_best.exists():
            shutil.copy2(previous_best, run.path / "checkpoints" / previous_best.name)
        logger.info("Resuming from epoch %d: %s", start_epoch, resume_path)
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        stochastic_seed = int(config["random_seeds"]["stochastic_bottleneck"]) + epoch
        torch.manual_seed(stochastic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stochastic_seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: list[float] = []
        extra_values: dict[str, list[float]] = {}
        epoch_code_counts: torch.Tensor | None = None
        for batch_index, batch in enumerate(
            _training_batches(train_loader, augmentation_generator)
        ):
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            with _autocast(device, config):
                loss, values = batch_loss(model, batch, config, device)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["train"].get("gradient_clip_norm", 5.0))
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running.append(values["loss"])
            for key, value in values.items():
                if key == "codebook_counts":
                    counts = torch.tensor(value, dtype=torch.float64)
                    epoch_code_counts = (
                        counts if epoch_code_counts is None else epoch_code_counts + counts
                    )
                    continue
                if key not in {"loss", "ce"}:
                    extra_values.setdefault(key, []).append(value)
        if is_autoencoder:
            tune_metrics = {"tune_mse": evaluate_reconstruction_mse(model, tune_loader, device)}
            current_metric = tune_metrics["tune_mse"]
        else:
            sample_count = 32 if config["model"].get("family") == "vib" else 1
            rng_devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                evaluation_seed = int(config["random_seeds"]["evaluation_order"]) + epoch
                torch.manual_seed(evaluation_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(evaluation_seed)
                tune_metrics, _ = evaluate_clean(
                    model, tune_loader, device, sample_count=sample_count
                )
            if sample_count > 1:
                deterministic_metrics, _ = evaluate_clean(model, tune_loader, device)
                tune_metrics["deterministic_mean_accuracy"] = deterministic_metrics[
                    "clean_accuracy"
                ]
            current_metric = tune_metrics["clean_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": sum(running) / max(1, len(running)),
            **tune_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update({key: sum(values) / max(1, len(values)) for key, values in extra_values.items()})
        if epoch_code_counts is not None:
            probabilities = epoch_code_counts / epoch_code_counts.sum().clamp_min(1.0)
            row["codebook_counts"] = epoch_code_counts.to(torch.int64).tolist()
            row["codebook_active_fraction"] = float((epoch_code_counts > 0).double().mean())
            row["codebook_perplexity"] = float(
                torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            )
            collapse_epochs = collapse_epochs + 1 if row["codebook_active_fraction"] < 0.10 else 0
            codebook_collapsed = codebook_collapsed or collapse_epochs >= 5
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.5f tune_metric=%.6f",
            epoch,
            row["train_loss"],
            current_metric,
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "config": config,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_loader_rng_state": (
                train_loader.generator.get_state() if train_loader.generator is not None else None
            ),
            "augmentation_rng_state": augmentation_generator.get_state(),
        }
        torch.save(checkpoint, run.path / "checkpoints" / "last.pt")
        improved = current_metric < best_metric if is_autoencoder else current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_name = "best_tune_mse.pt" if is_autoencoder else "best_tune_accuracy.pt"
            torch.save(checkpoint, run.path / "checkpoints" / best_name)
        save_frame(pd.DataFrame(history), run.path / "history.parquet")
        current_metric_path = "tune_reconstruction.json" if is_autoencoder else "tune_clean.json"
        current_metric_key = "mse" if is_autoencoder else "clean_accuracy"
        write_json(
            run.path / "metrics" / current_metric_path,
            {current_metric_key: best_metric, "epoch": best_epoch},
        )
    save_frame(pd.DataFrame(history), run.path / "history.parquet")
    tune_metric_name = "tune_reconstruction.json" if is_autoencoder else "tune_clean.json"
    tune_metric_key = "mse" if is_autoencoder else "clean_accuracy"
    write_json(
        run.path / "metrics" / tune_metric_name,
        {tune_metric_key: best_metric, "epoch": best_epoch},
    )
    final_metrics = {
        "status": "completed",
        "seed": seed,
        "dataset": config.get("data", {}).get("manifest"),
        "model": config["model"],
        ("best_tune_mse" if is_autoencoder else "best_tune_accuracy"): best_metric,
        "codebook_collapsed": codebook_collapsed,
    }
    final_path = run.complete(final_metrics)
    logger.info("Training completed: %s", final_path)
    return final_path
