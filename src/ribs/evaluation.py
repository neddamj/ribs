"""Checkpoint loading, latent extraction, and attack evaluation."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .artifacts import load_latents, save_frame, save_latents
from .attacks import _seeded_rng, input_pgd, latent_pgd, prequantization_latent_pgd
from .collisions import calibrate_collision_threshold, nearest_opposing, targeted_collision_attack
from .config import config_hash
from .data import ImagenetteDataset, manifest_hash, sha256_file
from .models import create_model
from .square_attack import square_attack
from .training import choose_device
from .utils import environment_info, seed_everything, write_json


def _new_evaluation_dir(run_dir: Path, name: str, evaluation_config: dict[str, Any]) -> Path:
    base_name = f"{name}-{config_hash(evaluation_config)}"
    attempt = 0
    while True:
        candidate = run_dir / "evaluations" / f"{base_name}-attempt{attempt}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            attempt += 1


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> str:
    return checkpoint or (
        "best_tune_mse.pt"
        if (run_dir / "checkpoints" / "best_tune_mse.pt").exists()
        else "best_tune_accuracy.pt"
    )


def checkpoint_config(run_dir: str | Path, checkpoint: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    path = run_dir / "checkpoints" / checkpoint
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["config"]


def load_model(run_dir: str | Path, checkpoint: str | None = None):
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    path = run_dir / "checkpoints" / checkpoint
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    model = create_model(config)
    model.load_state_dict(state["model"])
    device = choose_device(config)
    return model.to(device), config, device


def make_loader(config: dict[str, Any], split: str, batch_size: int | None = None) -> DataLoader:
    data = config["data"]
    dataset = ImagenetteDataset(
        data["root"],
        data["manifest"],
        split,
        int(data.get("image_size", 224)),
        data.get("size", "320px"),
    )

    def worker_init(worker_id: int) -> None:
        seed_everything(int(config.get("seed", 0)) + 3000 + worker_id, deterministic=False)

    return DataLoader(
        dataset,
        batch_size=batch_size or int(config["train"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
    )


@torch.no_grad()
def evaluate_clean_run(
    run_dir: str | Path,
    split: str = "final",
    checkpoint: str | None = None,
    max_samples: int | None = None,
) -> Path:
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    model, config, device = load_model(run_dir, checkpoint)
    if config["model"].get("family") == "autoencoder":
        raise ValueError(
            "Autoencoder clean accuracy requires evaluate-autoencoder and a reference run"
        )
    evaluation_config = {
        "kind": "clean",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "split": split,
        "max_samples": max_samples,
        "stochastic_samples": 32 if config["model"].get("family") == "vib" else 1,
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "clean", evaluation_config)
    loader = make_loader(config, split)
    rows = []
    seen = 0
    sample_count = 32 if config["model"].get("family") == "vib" else 1
    model.eval()
    for batch_number, batch in enumerate(loader):
        if max_samples is not None and seen >= max_samples:
            break
        take = (
            len(batch["label"])
            if max_samples is None
            else min(len(batch["label"]), max_samples - seen)
        )
        images = batch["image"][:take].to(device)
        labels = torch.as_tensor(batch["label"][:take], device=device)
        deterministic_logits = model(images, sample=False).logits
        logits = _predict_logits(
            model,
            images,
            sample_count,
            int(config["attack"].get("seed", 2025)) + batch_number,
        )
        predictions = logits.argmax(dim=-1)
        deterministic_predictions = deterministic_logits.argmax(dim=-1)
        for sample_id, label, prediction, deterministic_prediction in zip(
            batch["sample_id"][:take],
            labels.cpu(),
            predictions.cpu(),
            deterministic_predictions.cpu(),
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(label),
                    "prediction": int(prediction),
                    "correct": bool(label == prediction),
                    "deterministic_mean_prediction": int(deterministic_prediction),
                    "deterministic_mean_correct": bool(label == deterministic_prediction),
                }
            )
        seen += take
    path = evaluation_dir / f"clean_{split}.parquet"
    frame = pd.DataFrame(rows)
    save_frame(frame, path)
    write_json(
        path.with_suffix(".json"),
        {
            "clean_accuracy": float(frame.correct.mean()),
            "deterministic_mean_accuracy": float(frame.deterministic_mean_correct.mean()),
            "num_samples": len(rows),
            **evaluation_config,
            "environment": environment_info(),
        },
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path


@torch.no_grad()
def extract_latents(
    run_dir: str | Path, split: str = "final", checkpoint: str | None = None
) -> Path:
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    metadata_path = run_dir / "artifacts" / f"latents_{split}_metadata.json"
    extraction_metadata = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "split": split,
    }
    latent_path = run_dir / "artifacts" / f"latents_{split}.safetensors"
    if metadata_path.exists() and latent_path.exists():
        import json

        if json.loads(metadata_path.read_text()) != extraction_metadata:
            raise ValueError("Refusing to overwrite latents extracted from a different checkpoint")
        return latent_path
    model, config, device = load_model(run_dir, checkpoint)
    loader = make_loader(config, split)
    canonical, pre, labels, sample_ids = [], [], [], []
    code_indices = []
    logvars = []
    model.eval()
    for batch in loader:
        output = model(batch["image"].to(device), sample=False)
        canonical.append(output.canonical_latent.detach().cpu().flatten(1))
        pre.append(output.pre_bottleneck.detach().cpu().flatten(1))
        labels.extend(int(label) for label in batch["label"])
        sample_ids.extend(batch["sample_id"])
        if "code_indices" in output.metadata:
            code_indices.append(output.metadata["code_indices"].detach().cpu().flatten(1))
        if "logvar" in output.metadata:
            logvars.append(output.metadata["logvar"].detach().cpu().flatten(1))
    latent_dir = run_dir / "artifacts"
    save_latents(
        latent_dir / f"latents_{split}.safetensors",
        {"canonical_latent": torch.cat(canonical), "pre_bottleneck": torch.cat(pre)},
    )
    if code_indices:
        save_latents(
            latent_dir / f"codes_{split}.safetensors", {"code_indices": torch.cat(code_indices)}
        )
    if logvars:
        save_latents(
            latent_dir / f"vib_{split}.safetensors",
            {"mu": torch.cat(canonical), "logvar": torch.cat(logvars)},
        )
    index = pd.DataFrame(
        {"sample_id": sample_ids, "label": labels, "row_index": range(len(sample_ids))}
    )
    save_frame(index, latent_dir / f"latents_{split}_index.parquet")
    write_json(metadata_path, extraction_metadata)
    return latent_dir / f"latents_{split}.safetensors"


def _sample_count(config: dict[str, Any]) -> int:
    return (
        int(config["attack"].get("eot_samples", 1)) if config["model"].get("family") == "vib" else 1
    )


@torch.no_grad()
def _predict_logits(
    model: torch.nn.Module, images: torch.Tensor, samples: int, seed: int | None = None
) -> torch.Tensor:
    context = _seeded_rng(images, seed) if seed is not None else nullcontext()
    with context:
        if samples <= 1:
            return model(images, sample=False).logits
        probabilities = sum(
            torch.softmax(model(images, sample=True).logits.float(), dim=-1) for _ in range(samples)
        )
        return (probabilities / samples).clamp_min(1e-12).log()


def evaluate_attacks(
    run_dir: str | Path,
    split: str = "final",
    checkpoint: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    model, config, device = load_model(run_dir, checkpoint)
    evaluation_config = {
        "kind": "robustness",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "split": split,
        "max_samples": max_samples,
        "attack": config["attack"],
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "robustness", evaluation_config)
    loader = make_loader(
        config,
        split,
        batch_size=min(int(config["train"].get("batch_size", 128)), max_samples or 10**9),
    )
    epsilons = [float(value) for value in config["attack"]["input_epsilons"]]
    rhos = sorted({0.0, *(float(value) for value in config["attack"]["latent_rhos"])})
    attack_cfg = config["attack"]
    attack_seed = int(attack_cfg.get("seed", 2025))
    input_rows, latent_rows, latent_diagnostic_rows = [], [], []
    seen = 0
    for batch_number, batch in enumerate(loader):
        if max_samples is not None and seen >= max_samples:
            break
        take = (
            len(batch["label"])
            if max_samples is None
            else min(len(batch["label"]), max_samples - seen)
        )
        images = batch["image"][:take].to(device)
        labels = torch.as_tensor(batch["label"][:take], device=device)
        sample_ids = batch["sample_id"][:take]
        batch_seed = attack_seed + batch_number * 1_000_000
        sample_count = _sample_count(config)
        with torch.no_grad():
            clean_logits = _predict_logits(model, images, sample_count, batch_seed)
            clean_predictions = clean_logits.argmax(dim=-1)
        for epsilon in epsilons:
            radius_seed = batch_seed
            result = input_pgd(
                model,
                images,
                labels,
                epsilon,
                int(attack_cfg.get("steps", 40)),
                int(attack_cfg.get("restarts", 5)),
                sample_count,
                seed=radius_seed,
            )
            with torch.no_grad():
                adv_pred = (
                    clean_predictions
                    if epsilon == 0.0
                    else _predict_logits(
                        model, result.adversarial, sample_count, radius_seed + 5_000
                    ).argmax(dim=-1)
                )
                perturbation_norm = (result.adversarial - images).abs().flatten(1).amax(dim=1)
            for index, sample_id in enumerate(sample_ids):
                input_rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels[index]),
                        "radius": epsilon,
                        "clean_prediction": int(clean_predictions[index]),
                        "clean_correct": bool(clean_predictions[index] == labels[index]),
                        "adversarial_prediction": int(adv_pred[index]),
                        "successful": bool(adv_pred[index] != labels[index]),
                        "loss": float(result.loss[index]),
                        "initial_loss": float(result.initial_loss[index]),
                        "linf_norm": float(perturbation_norm[index]),
                        "best_restart": (
                            int(result.restart[index]) if result.restart is not None else -1
                        ),
                        "attack_seed": radius_seed,
                        "checkpoint": checkpoint,
                    }
                )
        family = config["model"].get("family", "dimensional")
        discrete = family in {"vq", "quantized", "quantized_continuous"}
        with torch.no_grad():
            clean_output = model(images, sample=False)
            clean_attack_latent = (
                clean_output.pre_bottleneck.detach().flatten(1)
                if discrete
                else clean_output.canonical_latent.detach()
            )
            latent_clean_logits = (
                model.classify_pre_bottleneck(clean_attack_latent)
                if discrete
                else model.classify_latent(clean_attack_latent)
            )
            latent_clean_predictions = latent_clean_logits.argmax(dim=-1)
            latent_clean_loss = torch.nn.functional.cross_entropy(
                latent_clean_logits.float(), labels, reduction="none"
            )
        for rho in rhos:
            attack_function = prequantization_latent_pgd if discrete else latent_pgd
            radius_seed = batch_seed + 200_000 + round(rho * 1000) * 100
            if rho == 0.0:
                adversarial_latent = clean_attack_latent
                result_loss = latent_clean_loss
                result_initial_loss = latent_clean_loss
                adv_pred = latent_clean_predictions
            else:
                result = attack_function(
                    model,
                    images,
                    labels,
                    rho,
                    int(attack_cfg.get("steps", 40)),
                    int(attack_cfg.get("restarts", 5)),
                    seed=radius_seed,
                )
                adversarial_latent = result.adversarial
                result_loss = result.loss
                result_initial_loss = result.initial_loss
                with torch.no_grad():
                    adv_logits = (
                        model.classify_pre_bottleneck(adversarial_latent)
                        if discrete
                        else model.classify_latent(adversarial_latent)
                    )
                    adv_pred = adv_logits.argmax(dim=-1)
            latent_delta_norm = (adversarial_latent - clean_attack_latent).flatten(1).norm(dim=1)
            latent_base_norm = clean_attack_latent.flatten(1).norm(dim=1).clamp_min(1e-12)
            target_rows = latent_rows
            surface = "pre_quantization" if discrete else "canonical_latent"
            for index, sample_id in enumerate(sample_ids):
                target_rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels[index]),
                        "radius": rho,
                        "clean_prediction": int(latent_clean_predictions[index]),
                        "clean_correct": bool(latent_clean_predictions[index] == labels[index]),
                        "adversarial_prediction": int(adv_pred[index]),
                        "successful": bool(adv_pred[index] != labels[index]),
                        "loss": float(result_loss[index]),
                        "initial_loss": float(result_initial_loss[index]),
                        "attack_surface": surface,
                        "l2_norm": float(latent_delta_norm[index]),
                        "relative_l2_norm": float(
                            latent_delta_norm[index] / latent_base_norm[index]
                        ),
                        "attack_seed": radius_seed,
                        "checkpoint": checkpoint,
                    }
                )
            if discrete and rho > 0.0:
                diagnostic = latent_pgd(
                    model,
                    images,
                    labels,
                    rho,
                    int(attack_cfg.get("steps", 40)),
                    int(attack_cfg.get("restarts", 5)),
                    seed=radius_seed + 100_000,
                )
                with torch.no_grad():
                    diagnostic_pred = model.classify_latent(diagnostic.adversarial).argmax(dim=-1)
                for index, sample_id in enumerate(sample_ids):
                    latent_diagnostic_rows.append(
                        {
                            "sample_id": sample_id,
                            "label": int(labels[index]),
                            "radius": rho,
                            "clean_prediction": int(latent_clean_predictions[index]),
                            "clean_correct": bool(latent_clean_predictions[index] == labels[index]),
                            "adversarial_prediction": int(diagnostic_pred[index]),
                            "successful": bool(diagnostic_pred[index] != labels[index]),
                            "loss": float(diagnostic.loss[index]),
                            "initial_loss": float(diagnostic.initial_loss[index]),
                            "attack_surface": "post_bottleneck_ambient",
                            "attack_seed": radius_seed + 100_000,
                            "checkpoint": checkpoint,
                        }
                    )
        seen += take
    paths = {
        "input": evaluation_dir / "input_pgd.parquet",
        "latent": evaluation_dir / "latent_pgd.parquet",
    }
    save_frame(pd.DataFrame(input_rows), paths["input"])
    save_frame(pd.DataFrame(latent_rows), paths["latent"])
    if latent_diagnostic_rows:
        paths["latent_diagnostic"] = evaluation_dir / "latent_pgd_post_bottleneck.parquet"
        save_frame(pd.DataFrame(latent_diagnostic_rows), paths["latent_diagnostic"])
    write_json(
        evaluation_dir / "config.json",
        {**evaluation_config, "environment": environment_info()},
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if max_samples is None and bool(attack_cfg.get("run_diagnostics", True)):
        paths["diagnostics"] = evaluate_attack_diagnostics(run_dir, split, checkpoint)
    return paths


def evaluate_attack_diagnostics(
    run_dir: str | Path,
    split: str = "final",
    checkpoint: str | None = None,
) -> Path:
    """Run prespecified step/restart and EoT convergence checks on a shared subset."""
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    model, config, device = load_model(run_dir, checkpoint)
    loader = make_loader(config, split, batch_size=1)
    dataset = loader.dataset
    sample_count = int(config["attack"].get("diagnostic_samples", 32))
    indices = _stratified_dataset_indices(dataset, sample_count)
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    labels = torch.tensor([sample["label"] for sample in samples], device=device)
    attack_cfg = config["attack"]
    steps = int(attack_cfg.get("steps", 40))
    restarts = int(attack_cfg.get("restarts", 5))
    eot_samples = _sample_count(config)
    seed = int(attack_cfg.get("seed", 2025)) + 90_000_000
    tolerance = float(attack_cfg.get("diagnostic_tolerance", 0.02))
    rows = []
    failures = []
    for epsilon in [float(value) for value in attack_cfg["input_epsilons"] if float(value) > 0]:
        baseline = input_pgd(model, images, labels, epsilon, steps, restarts, eot_samples, seed)
        stronger = input_pgd(
            model,
            images,
            labels,
            epsilon,
            steps * 2,
            restarts * 2,
            eot_samples,
            seed,
        )
        baseline_robust = float((~baseline.successful).float().mean())
        stronger_robust = float((~stronger.successful).float().mean())
        passed = stronger_robust <= baseline_robust + tolerance
        row = {
            "epsilon": epsilon,
            "baseline_robust_accuracy": baseline_robust,
            "stronger_robust_accuracy": stronger_robust,
            "passed": passed,
        }
        if config["model"].get("family") == "vib":
            with torch.no_grad():
                attacked_prediction = _predict_logits(
                    model, baseline.adversarial, eot_samples, seed + 1
                ).argmax(1)
                attacked_prediction_increased = _predict_logits(
                    model, baseline.adversarial, eot_samples * 2, seed + 1
                ).argmax(1)
            row["eot_attacked_prediction_disagreement"] = float(
                (attacked_prediction != attacked_prediction_increased).float().mean()
            )
            if row["eot_attacked_prediction_disagreement"] > tolerance:
                failures.append(
                    "attacked EoT prediction disagreement "
                    f"{row['eot_attacked_prediction_disagreement']:.4f} exceeds tolerance "
                    f"at epsilon={epsilon}"
                )
        rows.append(row)
        if not passed:
            failures.append(f"stronger PGD raised robust accuracy at epsilon={epsilon}")
    zero = input_pgd(model, images, labels, 0.0, steps, restarts, eot_samples, seed)
    if not torch.equal(zero.adversarial, images.float()):
        failures.append("epsilon=0 did not return the clean input exactly")
    if config["model"].get("family") == "vib":
        with torch.no_grad():
            base_prediction = _predict_logits(model, images, eot_samples, seed).argmax(1)
            increased_prediction = _predict_logits(model, images, eot_samples * 2, seed).argmax(1)
        disagreement = float((base_prediction != increased_prediction).float().mean())
        if disagreement > tolerance:
            failures.append(f"EoT prediction disagreement {disagreement:.4f} exceeds tolerance")
    else:
        disagreement = 0.0
    report = {
        "status": "passed" if not failures else "failed",
        "checkpoint": checkpoint,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "rows": rows,
        "eot_prediction_disagreement": disagreement,
        "tolerance": tolerance,
        "failures": failures,
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "attack-diagnostics", report)
    path = evaluation_dir / "diagnostics.json"
    write_json(path, report)
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Attack diagnostics failed: " + "; ".join(failures))
    return path


class _ReconstructionTask(torch.nn.Module):
    """Expose an autoencoder plus frozen classifier to the attack interface."""

    def __init__(self, autoencoder: torch.nn.Module, reference: torch.nn.Module):
        super().__init__()
        self.autoencoder = autoencoder
        self.reference = reference
        self.reference.requires_grad_(False)

    def forward(self, x: torch.Tensor, *, sample: bool = False):
        output = self.autoencoder(x, sample=False)
        reconstruction = output.metadata["reconstruction"]
        return type(
            "TaskOutput",
            (),
            {
                "logits": self.reference(reconstruction).logits,
                "latent": output.latent,
                "canonical_latent": output.canonical_latent,
                "pre_bottleneck": getattr(output, "pre_bottleneck", output.latent),
                "metadata": output.metadata,
            },
        )()

    def classify_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.reference(self.autoencoder.decode(z)).logits


@torch.no_grad()
def evaluate_autoencoder(
    run_dir: str | Path, reference_run_dir: str | Path, split: str = "final"
) -> Path:
    run_dir = Path(run_dir)
    autoencoder_checkpoint = _resolve_checkpoint(run_dir, None)
    reference_run_dir = Path(reference_run_dir)
    reference_checkpoint = _resolve_checkpoint(reference_run_dir, None)
    autoencoder, config, device = load_model(run_dir)
    reference, reference_config, reference_device = load_model(reference_run_dir)
    if reference_device != device:
        raise ValueError("Autoencoder and reference classifier must use the same device")
    if manifest_hash(config["data"]["manifest"]) != manifest_hash(
        reference_config["data"]["manifest"]
    ):
        raise ValueError("Autoencoder and reference classifier use different data manifests")
    evaluation_config = {
        "kind": "autoencoder_clean",
        "split": split,
        "checkpoint": autoencoder_checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / autoencoder_checkpoint),
        "reference_checkpoint": str(reference_run_dir / "checkpoints" / reference_checkpoint),
        "reference_checkpoint_sha256": sha256_file(
            reference_run_dir / "checkpoints" / reference_checkpoint
        ),
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "autoencoder-clean", evaluation_config)
    loader = make_loader(config, split)
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        with torch.no_grad():
            reconstruction = autoencoder(images).metadata["reconstruction"]
            original_logits = reference(images).logits
            reconstruction_logits = reference(reconstruction).logits
        for sample_id, label, original, reconstructed in zip(
            batch["sample_id"],
            batch["label"],
            original_logits.argmax(1).cpu(),
            reconstruction_logits.argmax(1).cpu(),
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(label),
                    "reference_original_prediction": int(original),
                    "reconstruction_prediction": int(reconstructed),
                    "reference_original_correct": bool(int(original) == int(label)),
                    "reconstruction_correct": bool(int(reconstructed) == int(label)),
                }
            )
    path = evaluation_dir / f"reconstruction_{split}.parquet"
    save_frame(pd.DataFrame(rows), path)
    write_json(
        path.with_suffix(".json"),
        {
            "reference_original_accuracy": float(
                pd.DataFrame(rows).reference_original_correct.mean()
            ),
            "reconstruction_accuracy": float(pd.DataFrame(rows).reconstruction_correct.mean()),
            **evaluation_config,
            "environment": environment_info(),
        },
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path


def evaluate_autoencoder_attacks(
    run_dir: str | Path,
    reference_run_dir: str | Path,
    split: str = "final",
    max_samples: int | None = None,
) -> dict[str, Path]:
    """Evaluate input and latent attacks through a frozen reference classifier."""
    run_dir = Path(run_dir)
    autoencoder_checkpoint = _resolve_checkpoint(run_dir, None)
    reference_run_dir = Path(reference_run_dir)
    reference_checkpoint = _resolve_checkpoint(reference_run_dir, None)
    autoencoder, config, device = load_model(run_dir)
    reference, reference_config, reference_device = load_model(reference_run_dir)
    if reference_device != device:
        raise ValueError("Autoencoder and reference classifier must use the same device")
    if manifest_hash(config["data"]["manifest"]) != manifest_hash(
        reference_config["data"]["manifest"]
    ):
        raise ValueError("Autoencoder and reference classifier use different data manifests")
    evaluation_config = {
        "kind": "autoencoder_robustness",
        "split": split,
        "max_samples": max_samples,
        "checkpoint": autoencoder_checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / autoencoder_checkpoint),
        "reference_checkpoint": str(reference_run_dir / "checkpoints" / reference_checkpoint),
        "reference_checkpoint_sha256": sha256_file(
            reference_run_dir / "checkpoints" / reference_checkpoint
        ),
        "attack": config["attack"],
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "autoencoder-robustness", evaluation_config)
    task = _ReconstructionTask(autoencoder, reference).to(device).eval()
    loader = make_loader(config, split)
    input_rows, latent_rows = [], []
    seen = 0
    attack_cfg = config["attack"]
    attack_seed = int(attack_cfg.get("seed", 2025))
    for batch_number, batch in enumerate(loader):
        if max_samples is not None and seen >= max_samples:
            break
        take = (
            len(batch["label"])
            if max_samples is None
            else min(len(batch["label"]), max_samples - seen)
        )
        images = batch["image"][:take].to(device)
        labels = torch.as_tensor(batch["label"][:take], device=device)
        sample_ids = batch["sample_id"][:take]
        batch_seed = attack_seed + batch_number * 1_000_000
        with torch.no_grad():
            clean_logits = task(images).logits
            clean_predictions = clean_logits.argmax(-1)
        for epsilon in [float(value) for value in attack_cfg["input_epsilons"]]:
            result = input_pgd(
                task,
                images,
                labels,
                epsilon,
                int(attack_cfg.get("steps", 40)),
                int(attack_cfg.get("restarts", 5)),
                seed=batch_seed + round(epsilon * 255) * 10_000,
            )
            with torch.no_grad():
                adversarial_prediction = task(result.adversarial).logits.argmax(-1)
            input_rows.extend(
                {
                    "sample_id": sample_id,
                    "label": int(labels[index]),
                    "radius": epsilon,
                    "clean_prediction": int(clean_predictions[index]),
                    "clean_correct": bool(clean_predictions[index] == labels[index]),
                    "adversarial_prediction": int(adversarial_prediction[index]),
                    "successful": bool(adversarial_prediction[index] != labels[index]),
                    "loss": float(result.loss[index]),
                    "initial_loss": float(result.initial_loss[index]),
                    "checkpoint": autoencoder_checkpoint,
                }
                for index, sample_id in enumerate(sample_ids)
            )
        with torch.no_grad():
            clean_latent = task(images).canonical_latent.detach()
            latent_clean_logits = task.classify_latent(clean_latent)
            latent_clean_prediction = latent_clean_logits.argmax(-1)
            latent_clean_loss = torch.nn.functional.cross_entropy(
                latent_clean_logits.float(), labels, reduction="none"
            )
        for rho in sorted({0.0, *(float(value) for value in attack_cfg["latent_rhos"])}):
            if rho == 0.0:
                adversarial_latent = clean_latent
                adversarial_prediction = latent_clean_prediction
                result_loss = latent_clean_loss
                result_initial_loss = latent_clean_loss
            else:
                result = latent_pgd(
                    task,
                    images,
                    labels,
                    rho,
                    int(attack_cfg.get("steps", 40)),
                    int(attack_cfg.get("restarts", 5)),
                    seed=batch_seed + 200_000 + round(rho * 1000) * 100,
                )
                adversarial_latent = result.adversarial
                result_loss = result.loss
                result_initial_loss = result.initial_loss
                with torch.no_grad():
                    adversarial_prediction = task.classify_latent(adversarial_latent).argmax(-1)
            latent_rows.extend(
                {
                    "sample_id": sample_id,
                    "label": int(labels[index]),
                    "radius": rho,
                    "clean_prediction": int(latent_clean_prediction[index]),
                    "clean_correct": bool(latent_clean_prediction[index] == labels[index]),
                    "adversarial_prediction": int(adversarial_prediction[index]),
                    "successful": bool(adversarial_prediction[index] != labels[index]),
                    "loss": float(result_loss[index]),
                    "initial_loss": float(result_initial_loss[index]),
                    "attack_surface": "canonical_latent",
                    "checkpoint": autoencoder_checkpoint,
                }
                for index, sample_id in enumerate(sample_ids)
            )
        seen += take
    paths = {
        "input": evaluation_dir / "input_pgd_reference.parquet",
        "latent": evaluation_dir / "latent_pgd_reference.parquet",
    }
    save_frame(pd.DataFrame(input_rows), paths["input"])
    save_frame(pd.DataFrame(latent_rows), paths["latent"])
    write_json(
        evaluation_dir / "config.json", {**evaluation_config, "environment": environment_info()}
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return paths


def _stratified_dataset_indices(dataset: ImagenetteDataset, count: int) -> list[int]:
    pools: dict[int, list[int]] = {}
    for class_label, group in dataset.frame.groupby("class_index", sort=True):
        pools[int(class_label)] = sorted(
            group.index.tolist(),
            key=lambda index: hashlib.sha256(
                str(dataset.frame.iloc[index].sample_id).encode()
            ).digest(),
        )
    selected: list[int] = []
    offset = 0
    while len(selected) < min(count, len(dataset)):
        added = False
        for class_label in sorted(pools):
            if offset < len(pools[class_label]) and len(selected) < count:
                selected.append(pools[class_label][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


@torch.no_grad()
def _eligibility_records(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    sample_ids: list[str] = []
    labels, eligible = [], []
    for batch in loader:
        images = batch["image"].to(device)
        current_labels = torch.as_tensor(batch["label"], device=device)
        model_prediction = model(images, sample=False).logits.argmax(1)
        reference_prediction = reference(images, sample=False).logits.argmax(1)
        sample_ids.extend(batch["sample_id"])
        labels.append(current_labels.cpu())
        eligible.append(
            ((model_prediction == current_labels) & (reference_prediction == current_labels)).cpu()
        )
    return sample_ids, torch.cat(labels), torch.cat(eligible)


def _candidate_collision_pairs(
    sample_ids: list[str], labels: torch.Tensor
) -> list[tuple[int, int]]:
    classes = sorted(int(value) for value in torch.unique(labels))
    source_order = sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(sample_ids[index].encode()).digest(),
    )
    pairs = []
    for source_index in source_order:
        for target_class in classes:
            if target_class == int(labels[source_index]):
                continue
            candidates = torch.where(labels == target_class)[0].tolist()
            target_index = min(
                candidates,
                key=lambda index: hashlib.sha256(
                    f"{sample_ids[source_index]}\0{sample_ids[index]}".encode()
                ).digest(),
            )
            pairs.append((source_index, target_index))
    return pairs


def evaluate_collision_attacks(
    run_dir: str | Path,
    reference_run_dir: str | Path,
    split: str = "final",
    max_pairs: int = 1000,
    lambda_sem: float | None = None,
) -> Path:
    run_dir = Path(run_dir)
    model, config, device = load_model(run_dir)
    lambda_override = lambda_sem is not None
    if lambda_sem is None:
        lambda_sem = float(config["attack"].get("collision_lambda_sem", 1.0))
    if split == "final" and lambda_override:
        raise ValueError("Final collision evaluation must use the frozen lambda_sem from config")
    if split == "final" and not bool(config["attack"].get("collision_lambda_sem_tuned", False)):
        raise ValueError(
            "Tune collision_lambda_sem on development_tune and set "
            "attack.collision_lambda_sem_tuned=true before final evaluation"
        )
    evaluation_config = {
        "kind": "collision",
        "split": split,
        "max_pairs": max_pairs,
        "lambda_sem": lambda_sem,
        "attack": config["attack"],
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "collision", evaluation_config)
    reference, reference_config, reference_device = load_model(reference_run_dir)
    if reference_device != device:
        raise ValueError(
            "Collision attack requires model and reference classifier on the same device"
        )
    if manifest_hash(config["data"]["manifest"]) != manifest_hash(
        reference_config["data"]["manifest"]
    ):
        raise ValueError("Collision model and reference classifier use different data manifests")
    if config["model"].get("family") == "autoencoder":
        model = _ReconstructionTask(model, reference).to(device).eval()
    model.eval()
    reference.eval()
    eligibility_loader = make_loader(
        config, split, batch_size=int(config["attack"].get("evaluation_batch_size", 32))
    )
    dataset = eligibility_loader.dataset
    sample_ids, labels, eligible = _eligibility_records(
        model, reference, eligibility_loader, device
    )
    candidate_pairs = _candidate_collision_pairs(sample_ids, labels)
    pairs = [
        pair for pair in candidate_pairs if bool(eligible[pair[0]]) and bool(eligible[pair[1]])
    ][:max_pairs]
    if not pairs:
        raise ValueError("No eligible source-target pairs for collision attack")
    tune_latent_path = run_dir / "artifacts" / "latents_development_tune.safetensors"
    tune_index_path = run_dir / "artifacts" / "latents_development_tune_index.parquet"
    if not tune_latent_path.exists():
        extract_latents(run_dir, "development_tune")
    tune = load_latents(tune_latent_path)
    tune_index = pd.read_parquet(tune_index_path)
    threshold = calibrate_collision_threshold(
        tune["canonical_latent"], torch.tensor(tune_index.label.to_numpy())
    )
    reference_distances, _ = nearest_opposing(
        tune["canonical_latent"], torch.tensor(tune_index.label.to_numpy())
    )
    pair_frame = pd.DataFrame(
        [
            {
                "source_sample_id": sample_ids[source],
                "target_sample_id": sample_ids[target],
                "source_label": int(labels[source]),
                "target_label": int(labels[target]),
            }
            for source, target in pairs
        ]
    )
    save_frame(pair_frame, evaluation_dir / "collision_pair_manifest.parquet")
    rows = []
    epsilons = [float(value) for value in config["attack"]["input_epsilons"] if float(value) > 0]
    for epsilon in epsilons:
        collision_batch_size = int(config["attack"].get("collision_batch_size", 8))
        for start in range(0, len(pairs), collision_batch_size):
            batch_pairs = pairs[start : start + collision_batch_size]
            source_indices = [pair[0] for pair in batch_pairs]
            target_indices = [pair[1] for pair in batch_pairs]
            source_images = torch.stack([dataset[index]["image"] for index in source_indices]).to(
                device
            )
            target_images = torch.stack([dataset[index]["image"] for index in target_indices]).to(
                device
            )
            source_labels = labels[source_indices].to(device)
            result = targeted_collision_attack(
                model,
                reference,
                source_images,
                source_labels,
                target_images,
                epsilon,
                threshold,
                int(config["attack"].get("collision_steps", 200)),
                int(config["attack"].get("collision_restarts", 5)),
                lambda_sem,
                seed=int(config["attack"].get("seed", 2025))
                + round(epsilon * 255) * 100_000
                + start,
            )
            with torch.no_grad():
                final_output = model(result["adversarial"], sample=False)
                target_output = model(target_images, sample=False)
                distances = (
                    F.normalize(final_output.canonical_latent.float().flatten(1), dim=1, eps=1e-12)
                    - F.normalize(
                        target_output.canonical_latent.float().flatten(1), dim=1, eps=1e-12
                    )
                ).norm(2, dim=1)
                source_prediction = reference(result["adversarial"]).logits.argmax(1)
                target_class_prediction = final_output.logits.argmax(1)
            for local, (source_index, target_index) in enumerate(batch_pairs):
                row = {
                    "source_sample_id": sample_ids[source_index],
                    "target_sample_id": sample_ids[target_index],
                    "source_label": int(labels[source_index]),
                    "target_label": int(labels[target_index]),
                    "epsilon": epsilon,
                    "distance": float(distances[local]),
                    "distance_percentile": float(
                        (reference_distances <= float(distances[local])).double().mean() * 100.0
                    ),
                    "reference_source_preserved": bool(
                        source_prediction[local] == source_labels[local]
                    ),
                    "target_class_prediction": int(target_class_prediction[local]),
                    "target_class_reached": bool(
                        target_class_prediction[local] == labels[target_index].to(device)
                    ),
                    "collision_success": bool(result["successful"][local]),
                    "collision_criterion": result["criterion"],
                }
                if result["criterion"] == "threshold":
                    row["continuous_collision"] = bool(distances[local] < threshold)
                if (
                    "code_indices" in final_output.metadata
                    and "code_indices" in target_output.metadata
                ):
                    row["exact_vq_collision"] = bool(
                        torch.equal(
                            final_output.metadata["code_indices"][local],
                            target_output.metadata["code_indices"][local],
                        )
                    )
                    row["vq_token_match_fraction"] = float(
                        (
                            final_output.metadata["code_indices"][local]
                            == target_output.metadata["code_indices"][local]
                        )
                        .float()
                        .mean()
                    )
                if (
                    "quantized_latent" in final_output.metadata
                    and "quantized_latent" in target_output.metadata
                ):
                    final_quantized = final_output.metadata["quantized_latent"][local]
                    target_quantized = target_output.metadata["quantized_latent"][local]
                    row["exact_quantized_collision"] = bool(
                        torch.equal(final_quantized, target_quantized)
                    )
                    row["quantized_bin_match_fraction"] = float(
                        (final_quantized == target_quantized).float().mean()
                    )
                rows.append(row)
    path = evaluation_dir / "collision_attacks.parquet"
    save_frame(pd.DataFrame(rows), path)
    write_json(
        path.with_suffix(".json"),
        {
            "threshold": threshold,
            "lambda_sem": lambda_sem,
            "num_pairs": len(pairs),
            **evaluation_config,
            "environment": environment_info(),
        },
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path


def evaluate_square_attack(
    run_dir: str | Path,
    split: str = "final",
    max_samples: int = 1000,
    checkpoint: str | None = None,
) -> Path:
    """Run the independent black-box check on a deterministic sample subset."""
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    model, config, device = load_model(run_dir, checkpoint)
    evaluation_config = {
        "kind": "square_attack",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "split": split,
        "max_samples": max_samples,
        "query_budget": config["attack"].get("square_queries", 5000),
        "attack_seed": config["attack"].get("seed", 2025),
    }
    evaluation_dir = _new_evaluation_dir(run_dir, "square", evaluation_config)
    loader = make_loader(config, split, batch_size=1)
    dataset = loader.dataset
    selected_indices = _stratified_dataset_indices(dataset, max_samples)
    selection_frame = dataset.frame.iloc[selected_indices][["sample_id", "class_index"]].rename(
        columns={"class_index": "label"}
    )
    selection_path = evaluation_dir / "square_attack_sample_manifest.parquet"
    save_frame(selection_frame.reset_index(drop=True), selection_path)
    rows = []
    evaluation_batch_size = int(config["attack"].get("square_batch_size", 16))
    attack_seed = int(config["attack"].get("seed", 2025))
    for batch_start in range(0, len(selected_indices), evaluation_batch_size):
        indices = selected_indices[batch_start : batch_start + evaluation_batch_size]
        samples = [dataset[index] for index in indices]
        images = torch.stack([sample["image"] for sample in samples]).to(device)
        labels = torch.tensor([sample["label"] for sample in samples], device=device)
        sample_ids = [str(sample["sample_id"]) for sample in samples]
        with torch.no_grad():
            clean_logits = _predict_logits(
                model, images, _sample_count(config), attack_seed + batch_start
            )
        for epsilon in [float(value) for value in config["attack"]["input_epsilons"]]:
            radius_seed = attack_seed + batch_start + round(epsilon * 255) * 100_000
            result = square_attack(
                model,
                images,
                labels,
                epsilon,
                int(config["attack"].get("square_queries", 5000)),
                radius_seed,
            )
            with torch.no_grad():
                adversarial_logits = (
                    clean_logits
                    if epsilon == 0.0
                    else _predict_logits(
                        model, result.adversarial, _sample_count(config), radius_seed + 50_000
                    )
                )
            for index, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels[index]),
                        "radius": epsilon,
                        "clean_prediction": int(clean_logits[index].argmax()),
                        "clean_correct": bool(clean_logits[index].argmax() == labels[index]),
                        "adversarial_prediction": int(adversarial_logits[index].argmax()),
                        "successful": bool(adversarial_logits[index].argmax() != labels[index]),
                        "loss": float(result.loss[index]),
                        "initial_loss": float(result.initial_loss[index]),
                        "queries": int(config["attack"].get("square_queries", 5000)),
                        "attack_seed": radius_seed,
                        "checkpoint": checkpoint,
                    }
                )
    path = evaluation_dir / "square_attack.parquet"
    save_frame(pd.DataFrame(rows), path)
    write_json(
        path.with_suffix(".json"),
        {**evaluation_config, "environment": environment_info()},
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path


def evaluate_transfer_attack(
    target_run_dir: str | Path,
    source_run_dir: str | Path,
    split: str = "final",
    max_samples: int = 1000,
) -> Path:
    """Attack a continuous source checkpoint and evaluate transfer to a target checkpoint."""
    target_run_dir = Path(target_run_dir)
    source_run_dir = Path(source_run_dir)
    target_checkpoint = _resolve_checkpoint(target_run_dir, None)
    source_checkpoint = _resolve_checkpoint(source_run_dir, None)
    target, target_config, target_device = load_model(target_run_dir, target_checkpoint)
    source, source_config, source_device = load_model(source_run_dir, source_checkpoint)
    if target_device != source_device:
        raise ValueError("Source and target models must use the same device")
    if manifest_hash(target_config["data"]["manifest"]) != manifest_hash(
        source_config["data"]["manifest"]
    ):
        raise ValueError("Transfer source and target use different data manifests")
    if source_config["model"].get("family") not in {"dimensional", "vib"}:
        raise ValueError("Transfer source must be a continuous dimensional or VIB model")
    loader = make_loader(target_config, split, batch_size=1)
    dataset = loader.dataset
    selected_indices = _stratified_dataset_indices(dataset, max_samples)
    evaluation_config = {
        "kind": "transfer_attack",
        "split": split,
        "max_samples": max_samples,
        "source_run": str(source_run_dir),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": sha256_file(source_run_dir / "checkpoints" / source_checkpoint),
        "target_checkpoint": target_checkpoint,
        "target_checkpoint_sha256": sha256_file(target_run_dir / "checkpoints" / target_checkpoint),
    }
    evaluation_dir = _new_evaluation_dir(target_run_dir, "transfer", evaluation_config)
    rows = []
    batch_size = int(target_config["attack"].get("evaluation_batch_size", 32))
    base_seed = int(target_config["attack"].get("seed", 2025))
    for start in range(0, len(selected_indices), batch_size):
        indices = selected_indices[start : start + batch_size]
        samples = [dataset[index] for index in indices]
        images = torch.stack([sample["image"] for sample in samples]).to(target_device)
        labels = torch.tensor([sample["label"] for sample in samples], device=target_device)
        with torch.no_grad():
            target_clean = target(images, sample=False).logits.argmax(1)
        for epsilon in [float(value) for value in target_config["attack"]["input_epsilons"]]:
            radius_seed = base_seed + start
            result = input_pgd(
                source,
                images,
                labels,
                epsilon,
                int(target_config["attack"].get("steps", 40)),
                int(target_config["attack"].get("restarts", 5)),
                _sample_count(source_config),
                radius_seed,
            )
            with torch.no_grad():
                target_adversarial = target(result.adversarial, sample=False).logits.argmax(1)
            for local, sample in enumerate(samples):
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "label": int(labels[local]),
                        "radius": epsilon,
                        "clean_prediction": int(target_clean[local]),
                        "clean_correct": bool(target_clean[local] == labels[local]),
                        "adversarial_prediction": int(target_adversarial[local]),
                        "successful": bool(target_adversarial[local] != labels[local]),
                        "linf_norm": float((result.adversarial[local] - images[local]).abs().max()),
                    }
                )
    path = evaluation_dir / "transfer_attack.parquet"
    save_frame(pd.DataFrame(rows), path)
    write_json(
        evaluation_dir / "config.json", {**evaluation_config, "environment": environment_info()}
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path
