"""Checkpoint loading, latent extraction, and attack evaluation."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .artifacts import load_latents, save_frame, save_latents
from .attacks import _seeded_rng, input_pgd, latent_pgd, prequantization_latent_pgd
from .collisions import calibrate_collision_threshold, nearest_opposing, targeted_collision_attack
from .config import config_hash
from .data import ImagenetteDataset, manifest_hash, sample_ids_hash, sha256_file
from .models import create_model
from .phase2 import load_attack_audit_amendment
from .square_attack import square_attack
from .training import choose_device
from .utils import environment_info, seed_everything, write_json

ATTACK_PROTOCOL_VERSION = 2


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


def _matching_evaluation_config(path: Path, expected: dict[str, Any]) -> bool:
    """Return whether an evaluation directory has the requested immutable identity."""
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        observed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(observed.get(key) == value for key, value in expected.items())


def _resumable_evaluation_dir(
    run_dir: Path, name: str, evaluation_config: dict[str, Any], result_name: str
) -> tuple[Path, bool]:
    """Reuse one exact completed/incomplete evaluation, or create a new attempt.

    Completed artifacts remain immutable. An incomplete directory is resumed only
    when its saved identity contains every requested configuration field. Multiple
    matching attempts are rejected rather than selected heuristically.
    """
    base_name = f"{name}-{config_hash(evaluation_config)}"
    matches = sorted(
        path
        for path in (run_dir / "evaluations").glob(f"{base_name}-attempt*")
        if _matching_evaluation_config(path, evaluation_config)
    )
    completed = [
        path for path in matches if (path / "COMPLETED").exists() and (path / result_name).exists()
    ]
    if len(completed) > 1:
        raise ValueError(f"Multiple completed evaluations match {name}: {completed}")
    if completed:
        return completed[0], True
    incomplete = [path for path in matches if not (path / "COMPLETED").exists()]
    if len(incomplete) > 1:
        raise ValueError(f"Multiple incomplete evaluations match {name}: {incomplete}")
    if incomplete:
        return incomplete[0], False
    path = _new_evaluation_dir(run_dir, name, evaluation_config)
    write_json(path / "config.json", {**evaluation_config, "environment": environment_info()})
    return path, False


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
    # Evaluation only differentiates with respect to inputs/latents. Freezing
    # parameters preserves those gradients while avoiding parameter-gradient work.
    model = model.to(device).eval().requires_grad_(False)
    return model, config, device


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

    num_workers = int(data.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=batch_size or int(config["train"].get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init,
    )


def _validate_reference_config(
    config: dict[str, Any], reference_config: dict[str, Any], *, context: str
) -> None:
    if manifest_hash(config["data"]["manifest"]) != manifest_hash(
        reference_config["data"]["manifest"]
    ):
        raise ValueError(f"{context} and reference classifier use different data manifests")
    if str(reference_config.get("model", {}).get("family", "")).lower() != "identity":
        raise ValueError(
            f"{context} requires an independently trained identity reference classifier"
        )
    if int(reference_config.get("seed", -1)) != 0:
        raise ValueError(f"{context} requires the prescribed seed-0 reference classifier")


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
    loader = make_loader(
        config, split, batch_size=int(config["attack"].get("evaluation_batch_size", 32))
    )
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
        images = batch["image"][:take].to(device, non_blocking=True)
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
    loader = make_loader(
        config, split, batch_size=int(config["attack"].get("evaluation_batch_size", 32))
    )
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
    # The clean baseline is evaluated before input_pgd, so establish inference
    # mode here rather than relying on the attack function to do it later.
    model.eval()
    evaluation_config = {
        "kind": "robustness",
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
        batch_size=min(
            int(config["attack"].get("evaluation_batch_size", 32)), max_samples or 10**9
        ),
    )
    epsilons = sorted({float(value) for value in config["attack"]["input_epsilons"]})
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
        images = batch["image"][:take].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"][:take], device=device)
        sample_ids = batch["sample_id"][:take]
        batch_seed = attack_seed + batch_number * 1_000_000
        sample_count = _sample_count(config)
        with torch.no_grad():
            clean_logits = _predict_logits(model, images, sample_count, batch_seed)
            clean_predictions = clean_logits.argmax(dim=-1)
        previous_input_candidate = images
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
                initial_adversarial=previous_input_candidate,
            )
            previous_input_candidate = result.adversarial.detach()
            with torch.no_grad():
                adv_pred = (
                    clean_predictions
                    if epsilon == 0.0
                    else _predict_logits(
                        model, result.adversarial, sample_count, radius_seed + 5_000
                    ).argmax(dim=-1)
                )
                perturbation_norm = (result.adversarial - images).abs().flatten(1).amax(dim=1)
            labels_cpu = labels.cpu()
            clean_predictions_cpu = clean_predictions.cpu()
            adv_pred_cpu = adv_pred.cpu()
            perturbation_cpu = perturbation_norm.cpu()
            result_loss_cpu = result.loss.cpu()
            retained_loss_cpu = result.retained_loss.cpu()
            initial_loss_cpu = result.initial_loss.cpu()
            restart_cpu = result.restart.cpu() if result.restart is not None else None
            for index, sample_id in enumerate(sample_ids):
                input_rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels_cpu[index]),
                        "radius": epsilon,
                        "clean_prediction": int(clean_predictions_cpu[index]),
                        "clean_correct": bool(clean_predictions_cpu[index] == labels_cpu[index]),
                        "adversarial_prediction": int(adv_pred_cpu[index]),
                        "successful": bool(adv_pred_cpu[index] != labels_cpu[index]),
                        "loss": float(result_loss_cpu[index]),
                        "retained_loss": float(retained_loss_cpu[index]),
                        "initial_loss": float(initial_loss_cpu[index]),
                        "linf_norm": float(perturbation_cpu[index]),
                        "best_restart": (
                            int(restart_cpu[index]) if restart_cpu is not None else -1
                        ),
                        "attack_seed": radius_seed,
                        "checkpoint": checkpoint,
                        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
        previous_latent_candidate = clean_attack_latent
        previous_ambient_candidate = getattr(clean_output, "latent", clean_attack_latent).detach()
        for rho in rhos:
            attack_function = prequantization_latent_pgd if discrete else latent_pgd
            radius_seed = batch_seed + 200_000 + round(rho * 1000) * 100
            if rho == 0.0:
                adversarial_latent = clean_attack_latent
                result_loss = latent_clean_loss
                result_retained_loss = latent_clean_loss
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
                    initial_adversarial=previous_latent_candidate,
                )
                previous_latent_candidate = result.adversarial.detach()
                adversarial_latent = result.adversarial
                result_loss = result.loss
                result_retained_loss = result.retained_loss
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
            labels_cpu = labels.cpu()
            latent_clean_predictions_cpu = latent_clean_predictions.cpu()
            adv_pred_cpu = adv_pred.cpu()
            result_loss_cpu = result_loss.cpu()
            result_retained_cpu = result_retained_loss.cpu()
            result_initial_cpu = result_initial_loss.cpu()
            latent_delta_cpu = latent_delta_norm.cpu()
            latent_base_cpu = latent_base_norm.cpu()
            restart_cpu = result.restart.cpu() if rho > 0.0 and result.restart is not None else None
            target_rows = latent_rows
            surface = "pre_quantization" if discrete else "canonical_latent"
            for index, sample_id in enumerate(sample_ids):
                target_rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels_cpu[index]),
                        "radius": rho,
                        "clean_prediction": int(latent_clean_predictions_cpu[index]),
                        "clean_correct": bool(
                            latent_clean_predictions_cpu[index] == labels_cpu[index]
                        ),
                        "adversarial_prediction": int(adv_pred_cpu[index]),
                        "successful": bool(adv_pred_cpu[index] != labels_cpu[index]),
                        "loss": float(result_loss_cpu[index]),
                        "retained_loss": float(result_retained_cpu[index]),
                        "initial_loss": float(result_initial_cpu[index]),
                        "attack_surface": surface,
                        "l2_norm": float(latent_delta_cpu[index]),
                        "relative_l2_norm": float(latent_delta_cpu[index] / latent_base_cpu[index]),
                        "attack_seed": radius_seed,
                        "best_restart": (
                            int(restart_cpu[index]) if restart_cpu is not None else -1
                        ),
                        "checkpoint": checkpoint,
                        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
                    initial_adversarial=previous_ambient_candidate,
                )
                previous_ambient_candidate = diagnostic.adversarial.detach()
                with torch.no_grad():
                    diagnostic_pred = model.classify_latent(diagnostic.adversarial).argmax(dim=-1)
                diagnostic_pred_cpu = diagnostic_pred.cpu()
                diagnostic_loss_cpu = diagnostic.loss.cpu()
                diagnostic_retained_cpu = diagnostic.retained_loss.cpu()
                diagnostic_initial_cpu = diagnostic.initial_loss.cpu()
                diagnostic_restart_cpu = (
                    diagnostic.restart.cpu() if diagnostic.restart is not None else None
                )
                diagnostic_delta = (diagnostic.adversarial - clean_output.latent).flatten(1)
                diagnostic_l2_cpu = diagnostic_delta.norm(dim=1).cpu()
                diagnostic_base_cpu = clean_output.latent.flatten(1).norm(dim=1).cpu()
                for index, sample_id in enumerate(sample_ids):
                    latent_diagnostic_rows.append(
                        {
                            "sample_id": sample_id,
                            "label": int(labels_cpu[index]),
                            "radius": rho,
                            "clean_prediction": int(latent_clean_predictions_cpu[index]),
                            "clean_correct": bool(
                                latent_clean_predictions_cpu[index] == labels_cpu[index]
                            ),
                            "adversarial_prediction": int(diagnostic_pred_cpu[index]),
                            "successful": bool(diagnostic_pred_cpu[index] != labels_cpu[index]),
                            "loss": float(diagnostic_loss_cpu[index]),
                            "retained_loss": float(diagnostic_retained_cpu[index]),
                            "initial_loss": float(diagnostic_initial_cpu[index]),
                            "attack_surface": "post_bottleneck_ambient",
                            "l2_norm": float(diagnostic_l2_cpu[index]),
                            "relative_l2_norm": float(
                                diagnostic_l2_cpu[index]
                                / diagnostic_base_cpu[index].clamp_min(1e-12)
                            ),
                            "best_restart": (
                                int(diagnostic_restart_cpu[index])
                                if diagnostic_restart_cpu is not None
                                else -1
                            ),
                            "attack_seed": radius_seed + 100_000,
                            "checkpoint": checkpoint,
                            "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
    diagnostic_samples: int | None = None,
    diagnostic_tolerance: float | None = None,
    audit_amendment: str | Path | None = None,
) -> Path:
    """Run convergence checks for every attack surface on a shared subset.

    Optional overrides are reserved for an independent convergence audit. They
    do not alter the frozen primary attack, and the resolved values are saved
    in the audit artifact for provenance.
    """
    run_dir = Path(run_dir)
    amendment = load_attack_audit_amendment(audit_amendment) if audit_amendment else None
    checkpoint = _resolve_checkpoint(run_dir, checkpoint)
    model, config, device = load_model(run_dir, checkpoint)
    loader = make_loader(config, split, batch_size=1)
    dataset = loader.dataset
    sample_count = int(
        diagnostic_samples
        if diagnostic_samples is not None
        else config["attack"].get("diagnostic_samples", 32)
    )
    if sample_count <= 0:
        raise ValueError("diagnostic_samples must be positive")
    indices = _stratified_dataset_indices(dataset, sample_count)
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    labels = torch.tensor([sample["label"] for sample in samples], device=device)
    attack_cfg = config["attack"]
    steps = int(attack_cfg.get("steps", 40))
    restarts = int(attack_cfg.get("restarts", 5))
    eot_samples = _sample_count(config)
    seed = int(attack_cfg.get("seed", 2025)) + 90_000_000
    tolerance = float(
        diagnostic_tolerance
        if diagnostic_tolerance is not None
        else attack_cfg.get("diagnostic_tolerance", 0.02)
    )
    if tolerance < 0:
        raise ValueError("diagnostic_tolerance must be non-negative")
    if amendment is not None:
        protocol = amendment["protocol"]
        if diagnostic_samples != int(protocol["sample_count"]):
            raise ValueError("Superseding audit must use the amended sample count")
        if abs(tolerance - float(protocol["tolerance"])) > 1e-12:
            raise ValueError("Superseding audit must retain the amended tolerance")
        if (steps, restarts) != (
            int(protocol["baseline"]["steps"]),
            int(protocol["baseline"]["restarts"]),
        ):
            raise ValueError("Superseding audit baseline does not match amendment")
    rows: list[dict[str, Any]] = []
    failures = []

    def compare(
        surface: str,
        radius: float,
        baseline,
        stronger,
        *,
        increased_eot=None,
    ) -> None:
        baseline_robust = float((~baseline.successful).float().mean())
        stronger_robust = float((~stronger.successful).float().mean())
        union_success = baseline.successful | stronger.successful
        union_robust = float((~union_success).float().mean())
        row = {
            "attack_surface": surface,
            "radius": radius,
            "baseline_robust_accuracy": baseline_robust,
            "stronger_robust_accuracy": stronger_robust,
            "union_robust_accuracy": union_robust,
            "stronger_shift": stronger_robust - baseline_robust,
            "convergence_drop": baseline_robust - union_robust,
        }
        passed = (
            stronger_robust <= baseline_robust + tolerance
            and baseline_robust - union_robust <= tolerance
        )
        if increased_eot is not None:
            increased_eot_robust = float((~increased_eot.successful).float().mean())
            eot_union_robust = float(
                (~(baseline.successful | increased_eot.successful)).float().mean()
            )
            row.update(
                {
                    "increased_eot_robust_accuracy": increased_eot_robust,
                    "eot_convergence_drop": baseline_robust - eot_union_robust,
                }
            )
            passed = passed and increased_eot_robust <= baseline_robust + tolerance
            passed = passed and baseline_robust - eot_union_robust <= tolerance
        row["passed"] = passed
        rows.append(row)
        if not passed:
            failures.append(
                f"attack convergence failed for surface={surface} radius={radius}: {row}"
            )
        if baseline.retained_loss is None or torch.any(
            baseline.retained_loss + 1e-6 < baseline.initial_loss
        ):
            failures.append(f"retained loss decreased for surface={surface} radius={radius}")

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
        increased_eot = None
        if config["model"].get("family") == "vib":
            increased_eot = input_pgd(
                model,
                images,
                labels,
                epsilon,
                steps,
                restarts,
                eot_samples * 2,
                seed,
            )
        compare("input", epsilon, baseline, stronger, increased_eot=increased_eot)
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

    family = str(config["model"].get("family", "dimensional")).lower()
    discrete = family in {"vq", "quantized", "quantized_continuous"}
    latent_attack = prequantization_latent_pgd if discrete else latent_pgd
    latent_surface = "pre_quantization" if discrete else "canonical_latent"
    for rho in [float(value) for value in attack_cfg["latent_rhos"] if float(value) > 0]:
        latent_seed = seed + 10_000_000 + round(rho * 1000)
        baseline = latent_attack(model, images, labels, rho, steps, restarts, latent_seed)
        stronger = latent_attack(model, images, labels, rho, steps * 2, restarts * 2, latent_seed)
        compare(latent_surface, rho, baseline, stronger)
        with torch.no_grad():
            output = model(images, sample=False)
            clean_latent = (
                output.pre_bottleneck.flatten(1) if discrete else output.canonical_latent.flatten(1)
            )
        allowed = rho * clean_latent.norm(dim=1).clamp_min(1e-12)
        observed = (baseline.adversarial.flatten(1) - clean_latent).norm(dim=1)
        if torch.any(observed > allowed + 1e-5):
            failures.append(f"L2 bound failed for surface={latent_surface} rho={rho}")
        if discrete:
            ambient_baseline = latent_pgd(
                model, images, labels, rho, steps, restarts, latent_seed + 1_000_000
            )
            ambient_stronger = latent_pgd(
                model,
                images,
                labels,
                rho,
                steps * 2,
                restarts * 2,
                latent_seed + 1_000_000,
            )
            compare("post_bottleneck_ambient", rho, ambient_baseline, ambient_stronger)
    report = {
        "status": "passed" if not failures else "failed",
        "checkpoint": checkpoint,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "rows": rows,
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        "baseline_steps": steps,
        "baseline_restarts": restarts,
        "baseline_eot_samples": eot_samples,
        "stronger_steps": steps * 2,
        "stronger_restarts": restarts * 2,
        "increased_eot_samples": eot_samples * 2 if family == "vib" else None,
        "eot_prediction_disagreement": disagreement,
        "diagnostic_samples": sample_count,
        "tolerance": tolerance,
        "independent_audit": diagnostic_samples is not None or diagnostic_tolerance is not None,
        "failures": failures,
    }
    if amendment is not None:
        manifest_path = Path(config["data"]["manifest"])
        report.update(
            {
                "audit_amendment_id": amendment["amendment_id"],
                "audit_amendment_sha256": sha256_file(Path(audit_amendment)),
                "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
                "data_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
                "sample_manifest_hash": sample_ids_hash(report["sample_ids"]),
                "sample_selection": amendment["protocol"]["sample_selection"],
            }
        )
    evaluation_dir = _new_evaluation_dir(run_dir, "attack-diagnostics", report)
    path = evaluation_dir / "diagnostics.json"
    write_json(path, report)
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Attack diagnostics failed: " + "; ".join(failures))
    return path


def evaluate_autoencoder_attack_diagnostics(
    run_dir: str | Path,
    reference_run_dir: str | Path,
    split: str = "final",
    checkpoint: str | None = None,
    diagnostic_samples: int | None = None,
    diagnostic_tolerance: float | None = None,
    audit_amendment: str | Path | None = None,
) -> Path:
    """Run PGD correctness checks on the autoencoder task wrapper.

    Optional overrides create a separately identified convergence audit and do
    not change the frozen primary attack settings.
    """
    run_dir = Path(run_dir)
    amendment = load_attack_audit_amendment(audit_amendment) if audit_amendment else None
    autoencoder, config, device = load_model(run_dir, checkpoint)
    reference, reference_config, reference_device = load_model(reference_run_dir)
    if device != reference_device:
        raise ValueError("Autoencoder and reference classifier must use the same device")
    _validate_reference_config(config, reference_config, context="Autoencoder attack diagnostics")
    task = _ReconstructionTask(autoencoder, reference).to(device).eval()
    loader = make_loader(config, split, batch_size=1)
    count = int(
        diagnostic_samples
        if diagnostic_samples is not None
        else config["attack"].get("diagnostic_samples", 32)
    )
    if count <= 0:
        raise ValueError("diagnostic_samples must be positive")
    indices = _stratified_dataset_indices(loader.dataset, count)
    samples = [loader.dataset[index] for index in indices]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    labels = torch.tensor([sample["label"] for sample in samples], device=device)
    attack_cfg = config["attack"]
    steps = int(attack_cfg.get("steps", 40))
    restarts = int(attack_cfg.get("restarts", 5))
    tolerance = float(
        diagnostic_tolerance
        if diagnostic_tolerance is not None
        else attack_cfg.get("diagnostic_tolerance", 0.02)
    )
    if tolerance < 0:
        raise ValueError("diagnostic_tolerance must be non-negative")
    if amendment is not None:
        protocol = amendment["protocol"]
        if diagnostic_samples != int(protocol["sample_count"]):
            raise ValueError("Superseding audit must use the amended sample count")
        if abs(tolerance - float(protocol["tolerance"])) > 1e-12:
            raise ValueError("Superseding audit must retain the amended tolerance")
    seed = int(attack_cfg.get("seed", 2025)) + 91_000_000
    failures = []
    rows = []
    for epsilon in [float(value) for value in attack_cfg["input_epsilons"] if float(value) > 0]:
        baseline = input_pgd(task, images, labels, epsilon, steps, restarts, seed=seed)
        stronger = input_pgd(task, images, labels, epsilon, steps * 2, restarts * 2, seed=seed)
        baseline_robust = float((~baseline.successful).float().mean())
        stronger_robust = float((~stronger.successful).float().mean())
        union_robust = float((~(baseline.successful | stronger.successful)).float().mean())
        passed = (
            stronger_robust <= baseline_robust + tolerance
            and baseline_robust - union_robust <= tolerance
        )
        rows.append(
            {
                "attack_surface": "input",
                "radius": epsilon,
                "baseline_robust_accuracy": baseline_robust,
                "stronger_robust_accuracy": stronger_robust,
                "union_robust_accuracy": union_robust,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"stronger PGD raised robust accuracy at epsilon={epsilon}")
        if torch.any(baseline.retained_loss + 1e-6 < baseline.initial_loss):
            failures.append(f"retained loss decreased at epsilon={epsilon}")
        if torch.any((baseline.adversarial - images).abs().flatten(1).amax(1) > epsilon + 1e-6):
            failures.append(f"L-infinity bound failed at epsilon={epsilon}")
    zero = input_pgd(task, images, labels, 0.0, steps, restarts, seed=seed)
    if not torch.equal(zero.adversarial, images.float()):
        failures.append("epsilon=0 did not return the clean input exactly")
    for rho in [float(value) for value in attack_cfg["latent_rhos"] if float(value) > 0]:
        latent_seed = seed + 10_000_000 + round(rho * 1000)
        baseline = latent_pgd(task, images, labels, rho, steps, restarts, latent_seed)
        stronger = latent_pgd(task, images, labels, rho, steps * 2, restarts * 2, latent_seed)
        baseline_robust = float((~baseline.successful).float().mean())
        stronger_robust = float((~stronger.successful).float().mean())
        union_robust = float((~(baseline.successful | stronger.successful)).float().mean())
        passed = (
            stronger_robust <= baseline_robust + tolerance
            and baseline_robust - union_robust <= tolerance
        )
        rows.append(
            {
                "attack_surface": "canonical_latent",
                "radius": rho,
                "baseline_robust_accuracy": baseline_robust,
                "stronger_robust_accuracy": stronger_robust,
                "union_robust_accuracy": union_robust,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"latent attack convergence failed at rho={rho}")
        if torch.any(baseline.retained_loss + 1e-6 < baseline.initial_loss):
            failures.append(f"retained loss decreased at rho={rho}")
        with torch.no_grad():
            clean_latent = task(images).canonical_latent.flatten(1)
        allowed = rho * clean_latent.norm(dim=1).clamp_min(1e-12)
        observed = (baseline.adversarial.flatten(1) - clean_latent).norm(dim=1)
        if torch.any(observed > allowed + 1e-5):
            failures.append(f"latent L2 bound failed at rho={rho}")
    report = {
        "status": "passed" if not failures else "failed",
        "checkpoint": checkpoint or _resolve_checkpoint(run_dir, None),
        "reference_run_dir": str(reference_run_dir),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "rows": rows,
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        "diagnostic_samples": count,
        "tolerance": tolerance,
        "independent_audit": diagnostic_samples is not None or diagnostic_tolerance is not None,
        "failures": failures,
    }
    if amendment is not None:
        manifest_path = Path(config["data"]["manifest"])
        checkpoint_name = checkpoint or _resolve_checkpoint(run_dir, None)
        report.update(
            {
                "audit_amendment_id": amendment["amendment_id"],
                "audit_amendment_sha256": sha256_file(Path(audit_amendment)),
                "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
                "data_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint_name),
                "sample_manifest_hash": sample_ids_hash(report["sample_ids"]),
                "sample_selection": amendment["protocol"]["sample_selection"],
            }
        )
    evaluation_dir = _new_evaluation_dir(run_dir, "autoencoder-attack-diagnostics", report)
    path = evaluation_dir / "diagnostics.json"
    write_json(path, report)
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Autoencoder attack diagnostics failed: " + "; ".join(failures))
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
    _validate_reference_config(config, reference_config, context="Autoencoder evaluation")
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
    loader = make_loader(
        config, split, batch_size=int(config["attack"].get("evaluation_batch_size", 32))
    )
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            reconstruction = autoencoder(images).metadata["reconstruction"]
            original_logits = reference(images).logits
            reconstruction_logits = reference(reconstruction).logits
            per_sample_mse = (reconstruction - images).square().flatten(1).mean(dim=1)
            per_sample_psnr = -10.0 * torch.log10(per_sample_mse.clamp_min(1e-12))
        for sample_id, label, original, reconstructed, mse, psnr in zip(
            batch["sample_id"],
            batch["label"],
            original_logits.argmax(1).cpu(),
            reconstruction_logits.argmax(1).cpu(),
            per_sample_mse.cpu(),
            per_sample_psnr.cpu(),
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(label),
                    "reference_original_prediction": int(original),
                    "reconstruction_prediction": int(reconstructed),
                    "reference_original_correct": bool(int(original) == int(label)),
                    "reconstruction_correct": bool(int(reconstructed) == int(label)),
                    "reconstruction_mse": float(mse),
                    "reconstruction_psnr_db": float(psnr),
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
            "reconstruction_mse": float(pd.DataFrame(rows).reconstruction_mse.mean()),
            "reconstruction_psnr_db": float(pd.DataFrame(rows).reconstruction_psnr_db.mean()),
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
    _validate_reference_config(config, reference_config, context="Autoencoder attack")
    evaluation_config = {
        "kind": "autoencoder_robustness",
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
    loader = make_loader(
        config,
        split,
        batch_size=min(
            int(config["attack"].get("evaluation_batch_size", 32)), max_samples or 10**9
        ),
    )
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
        previous_input_candidate = images
        for epsilon in sorted({float(value) for value in attack_cfg["input_epsilons"]}):
            result = input_pgd(
                task,
                images,
                labels,
                epsilon,
                int(attack_cfg.get("steps", 40)),
                int(attack_cfg.get("restarts", 5)),
                seed=batch_seed + round(epsilon * 255) * 10_000,
                initial_adversarial=previous_input_candidate,
            )
            previous_input_candidate = result.adversarial.detach()
            with torch.no_grad():
                adversarial_prediction = task(result.adversarial).logits.argmax(-1)
                linf_norm = (result.adversarial - images).abs().flatten(1).amax(1)
            labels_cpu = labels.cpu()
            clean_predictions_cpu = clean_predictions.cpu()
            adversarial_prediction_cpu = adversarial_prediction.cpu()
            result_loss_cpu = result.loss.cpu()
            result_retained_cpu = result.retained_loss.cpu()
            result_initial_cpu = result.initial_loss.cpu()
            linf_cpu = linf_norm.cpu()
            restart_cpu = result.restart.cpu() if result.restart is not None else None
            input_rows.extend(
                {
                    "sample_id": sample_id,
                    "label": int(labels_cpu[index]),
                    "radius": epsilon,
                    "clean_prediction": int(clean_predictions_cpu[index]),
                    "clean_correct": bool(clean_predictions_cpu[index] == labels_cpu[index]),
                    "adversarial_prediction": int(adversarial_prediction_cpu[index]),
                    "successful": bool(adversarial_prediction_cpu[index] != labels_cpu[index]),
                    "loss": float(result_loss_cpu[index]),
                    "retained_loss": float(result_retained_cpu[index]),
                    "initial_loss": float(result_initial_cpu[index]),
                    "linf_norm": float(linf_cpu[index]),
                    "best_restart": (int(restart_cpu[index]) if restart_cpu is not None else -1),
                    "attack_seed": batch_seed + round(epsilon * 255) * 10_000,
                    "checkpoint": autoencoder_checkpoint,
                    "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
        previous_latent_candidate = clean_latent
        for rho in sorted({0.0, *(float(value) for value in attack_cfg["latent_rhos"])}):
            if rho == 0.0:
                adversarial_latent = clean_latent
                adversarial_prediction = latent_clean_prediction
                result_loss = latent_clean_loss
                result_retained_loss = latent_clean_loss
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
                    initial_adversarial=previous_latent_candidate,
                )
                previous_latent_candidate = result.adversarial.detach()
                adversarial_latent = result.adversarial
                result_loss = result.loss
                result_retained_loss = result.retained_loss
                result_initial_loss = result.initial_loss
                with torch.no_grad():
                    adversarial_prediction = task.classify_latent(adversarial_latent).argmax(-1)
            labels_cpu = labels.cpu()
            latent_clean_cpu = latent_clean_prediction.cpu()
            adversarial_prediction_cpu = adversarial_prediction.cpu()
            result_loss_cpu = result_loss.cpu()
            result_retained_cpu = result_retained_loss.cpu()
            result_initial_cpu = result_initial_loss.cpu()
            latent_delta_cpu = (adversarial_latent - clean_latent).flatten(1).norm(dim=1).cpu()
            latent_base_cpu = clean_latent.flatten(1).norm(dim=1).cpu()
            restart_cpu = result.restart.cpu() if rho > 0.0 and result.restart is not None else None
            latent_rows.extend(
                {
                    "sample_id": sample_id,
                    "label": int(labels_cpu[index]),
                    "radius": rho,
                    "clean_prediction": int(latent_clean_cpu[index]),
                    "clean_correct": bool(latent_clean_cpu[index] == labels_cpu[index]),
                    "adversarial_prediction": int(adversarial_prediction_cpu[index]),
                    "successful": bool(adversarial_prediction_cpu[index] != labels_cpu[index]),
                    "loss": float(result_loss_cpu[index]),
                    "retained_loss": float(result_retained_cpu[index]),
                    "initial_loss": float(result_initial_cpu[index]),
                    "attack_surface": "canonical_latent",
                    "l2_norm": float(latent_delta_cpu[index]),
                    "relative_l2_norm": float(
                        latent_delta_cpu[index] / latent_base_cpu[index].clamp_min(1e-12)
                    ),
                    "best_restart": (int(restart_cpu[index]) if restart_cpu is not None else -1),
                    "attack_seed": batch_seed + 200_000 + round(rho * 1000) * 100,
                    "checkpoint": autoencoder_checkpoint,
                    "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
    if max_samples is None and bool(attack_cfg.get("run_diagnostics", True)):
        paths["diagnostics"] = evaluate_autoencoder_attack_diagnostics(
            run_dir, reference_run_dir, split, autoencoder_checkpoint
        )
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
        images = batch["image"].to(device, non_blocking=True)
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
    """Return deterministic pairs with broad source coverage first.

    Each pass assigns one opposing class to every source before assigning a
    second target to any source. This makes a capped experiment use many
    independent sources instead of nine correlated pairs for each early source.
    """
    classes = sorted(int(value) for value in torch.unique(labels))
    class_position = {label: index for index, label in enumerate(classes)}
    source_order = sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(sample_ids[index].encode()).digest(),
    )
    pairs = []
    for class_offset in range(1, len(classes)):
        for source_index in source_order:
            source_class = int(labels[source_index])
            target_class = classes[(class_position[source_class] + class_offset) % len(classes)]
            candidates = torch.where(labels == target_class)[0].tolist()
            target_index = min(
                candidates,
                key=lambda index: hashlib.sha256(
                    f"{sample_ids[source_index]}\0{sample_ids[index]}".encode()
                ).digest(),
            )
            pairs.append((source_index, target_index))
    return pairs


def _collision_batch_sizes(
    attack_config: dict[str, Any], family: str | None = None
) -> tuple[int, int]:
    """Return effective attack and eligibility batches without changing pairs."""
    collision_batch_size = int(attack_config.get("collision_batch_size", 8))
    evaluation_batch_size = int(attack_config.get("evaluation_batch_size", 32))
    if collision_batch_size < 1 or evaluation_batch_size < 1:
        raise ValueError("collision_batch_size and evaluation_batch_size must be positive")
    # The reconstruction graph for an autoencoder is much larger than the
    # classifier-only graph used by the other collision families.  Cap its
    # attack batch at two pairs while retaining any stricter configured limit.
    effective_attack_batch_size = (
        min(2, collision_batch_size) if family == "autoencoder" else collision_batch_size
    )
    # Eligibility runs under no_grad and does not retain the reconstruction graph,
    # so it can use the ordinary evaluation batch independently of attack memory.
    return effective_attack_batch_size, evaluation_batch_size


def _cache_collision_images(
    dataset: ImagenetteDataset, pairs: list[tuple[int, int]]
) -> dict[int, torch.Tensor]:
    """Decode each deterministic evaluation image used by collision pairs once."""
    selected_indices = sorted({index for pair in pairs for index in pair})
    return {index: dataset[index]["image"] for index in selected_indices}


def select_collision_lambda(metrics: pd.DataFrame) -> dict[str, Any]:
    """Select a tuning candidate without consulting the final split."""
    required = {
        "lambda_sem",
        "collision_success_rate",
        "reference_source_preservation_rate",
        "median_distance",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Collision tuning metrics are missing columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("Collision tuning requires at least one lambda candidate")
    ranked = metrics.sort_values(
        [
            "collision_success_rate",
            "reference_source_preservation_rate",
            "median_distance",
            "lambda_sem",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    selected = ranked.iloc[0]
    return {
        "selected_lambda_sem": float(selected.lambda_sem),
        "selection_rule": (
            "maximize collision success; then maximize source-label preservation; "
            "then minimize median representation distance; then choose the smaller lambda"
        ),
        "selected_metrics": {
            column: float(selected[column])
            for column in (
                "collision_success_rate",
                "reference_source_preservation_rate",
                "median_distance",
            )
        },
    }


def tune_collision_lambda(
    run_dir: str | Path,
    reference_run_dir: str | Path,
    lambdas: list[float],
    max_pairs: int = 200,
) -> Path:
    """Tune the semantic-loss weight on development_tune and freeze the result."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    run_dir = Path(run_dir)
    reference_run_dir = Path(reference_run_dir)
    candidates = sorted({float(value) for value in lambdas})
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError(
            "Collision lambda candidates must be a non-empty list of nonnegative values"
        )
    checkpoint = _resolve_checkpoint(run_dir, None)
    reference_checkpoint = _resolve_checkpoint(reference_run_dir, None)
    _, config, _ = load_model(run_dir, checkpoint)
    _, reference_config, _ = load_model(reference_run_dir, reference_checkpoint)
    _validate_reference_config(config, reference_config, context="Collision lambda tuning")
    tuning_config = {
        "kind": "collision_lambda_tuning",
        "split": "development_tune",
        "lambda_candidates": candidates,
        "max_pairs": int(max_pairs),
        "pair_selection": "deterministic_source_round_robin_v2",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "reference_run": str(reference_run_dir),
        "reference_checkpoint": reference_checkpoint,
        "reference_checkpoint_sha256": sha256_file(
            reference_run_dir / "checkpoints" / reference_checkpoint
        ),
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
    }
    tuning_dir, tuning_complete = _resumable_evaluation_dir(
        run_dir, "collision-tuning", tuning_config, "selection.json"
    )
    if tuning_complete:
        return tuning_dir / "selection.json"
    rows = []
    for candidate in candidates:
        result_path = evaluate_collision_attacks(
            run_dir,
            reference_run_dir,
            split="development_tune",
            max_pairs=max_pairs,
            lambda_sem=candidate,
        )
        records = pd.read_parquet(result_path)
        positive_radius = records[records.epsilon > 0]
        scored = positive_radius if not positive_radius.empty else records
        rows.append(
            {
                "lambda_sem": candidate,
                "collision_success_rate": float(scored.collision_success.astype(bool).mean()),
                "reference_source_preservation_rate": float(
                    scored.reference_source_preserved.astype(bool).mean()
                ),
                "median_distance": float(scored.distance.median()),
                "num_records": len(scored),
                "evaluation_dir": str(result_path.parent),
            }
        )
    metrics = pd.DataFrame(rows)
    sweep_path = tuning_dir / "lambda_sweep.parquet"
    save_frame(metrics, sweep_path)
    selection = {
        **select_collision_lambda(metrics),
        **tuning_config,
        "lambda_sweep_sha256": sha256_file(sweep_path),
    }
    selection_path = tuning_dir / "selection.json"
    write_json(selection_path, selection)
    (tuning_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return selection_path


def evaluate_collision_attacks(
    run_dir: str | Path,
    reference_run_dir: str | Path,
    split: str = "final",
    max_pairs: int = 1000,
    lambda_sem: float | None = None,
    tuning_artifact: str | Path | None = None,
) -> Path:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    run_dir = Path(run_dir)
    reference_run_dir = Path(reference_run_dir)
    checkpoint = _resolve_checkpoint(run_dir, None)
    reference_checkpoint = _resolve_checkpoint(reference_run_dir, None)
    model, config, device = load_model(run_dir, checkpoint)
    reference, reference_config, reference_device = load_model(
        reference_run_dir, reference_checkpoint
    )
    if reference_device != device:
        raise ValueError(
            "Collision attack requires model and reference classifier on the same device"
        )
    _validate_reference_config(config, reference_config, context="Collision attack")
    lambda_override = lambda_sem is not None
    if split == "final" and lambda_override:
        raise ValueError("Final collision evaluation must use a frozen tuning artifact")
    tuning_path: Path | None = None
    if split == "final":
        if tuning_artifact is None:
            raise ValueError(
                "Final collision evaluation requires --tuning-artifact from tune-collision"
            )
        tuning_path = Path(tuning_artifact)
        if tuning_path.is_dir():
            tuning_path = tuning_path / "selection.json"
        if not tuning_path.exists() or not (tuning_path.parent / "COMPLETED").exists():
            raise ValueError(f"Collision tuning artifact is incomplete: {tuning_path}")
        selection = json.loads(tuning_path.read_text(encoding="utf-8"))
        sweep_path = tuning_path.parent / "lambda_sweep.parquet"
        if not sweep_path.exists() or selection.get("lambda_sweep_sha256") != sha256_file(
            sweep_path
        ):
            raise ValueError("Collision tuning sweep is missing or its hash does not match")
        recomputed_selection = select_collision_lambda(pd.read_parquet(sweep_path))
        if float(selection.get("selected_lambda_sem", -1)) != float(
            recomputed_selection["selected_lambda_sem"]
        ):
            raise ValueError("Collision tuning selection does not match the registered rule")
        expected = {
            "kind": "collision_lambda_tuning",
            "split": "development_tune",
            "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
            "reference_checkpoint_sha256": sha256_file(
                reference_run_dir / "checkpoints" / reference_checkpoint
            ),
            "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        }
        mismatches = {
            key: (selection.get(key), value)
            for key, value in expected.items()
            if selection.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Collision tuning artifact does not match this run: {mismatches}")
        lambda_sem = float(selection["selected_lambda_sem"])
    elif lambda_sem is None:
        raise ValueError("Non-final collision evaluation requires an explicit lambda_sem")
    family = config["model"].get("family")
    collision_batch_size, eligibility_batch_size = _collision_batch_sizes(config["attack"], family)
    evaluation_config = {
        "kind": "collision",
        "split": split,
        "max_pairs": max_pairs,
        "pair_selection": "deterministic_source_round_robin_v2",
        "lambda_sem": lambda_sem,
        "collision_batch_size": int(config["attack"].get("collision_batch_size", 8)),
        "attack": config["attack"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "reference_run": str(reference_run_dir),
        "reference_checkpoint": reference_checkpoint,
        "reference_checkpoint_sha256": sha256_file(
            reference_run_dir / "checkpoints" / reference_checkpoint
        ),
        "tuning_artifact": str(tuning_path) if tuning_path is not None else None,
        "tuning_artifact_sha256": (sha256_file(tuning_path) if tuning_path is not None else None),
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
    }
    # Autoencoder shard boundaries and attack seeds depend on this effective
    # batch. Include it in the resumable identity so batch-1 shards are never
    # mixed with batch-2 shards after a restart.
    if family == "autoencoder":
        evaluation_config["collision_attack_batch_size"] = collision_batch_size
        evaluation_config["collision_eligibility_batch_size"] = eligibility_batch_size
    evaluation_dir, evaluation_complete = _resumable_evaluation_dir(
        run_dir, "collision", evaluation_config, "collision_attacks.parquet"
    )
    if evaluation_complete:
        return evaluation_dir / "collision_attacks.parquet"
    if family == "autoencoder":
        model = _ReconstructionTask(model, reference).to(device).eval().requires_grad_(False)
    model.eval()
    reference.eval()
    evaluation_config["collision_attack_batch_size"] = collision_batch_size
    evaluation_config["collision_eligibility_batch_size"] = eligibility_batch_size
    eligibility_loader = make_loader(config, split, batch_size=eligibility_batch_size)
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
    extract_latents(run_dir, "development_tune", checkpoint)
    tune = load_latents(tune_latent_path)
    tune_index = pd.read_parquet(tune_index_path)
    threshold = calibrate_collision_threshold(
        tune["canonical_latent"], torch.tensor(tune_index.label.to_numpy())
    )
    reference_distances, _ = nearest_opposing(
        tune["canonical_latent"], torch.tensor(tune_index.label.to_numpy())
    )
    sorted_reference_distances = torch.sort(reference_distances).values
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
    # Evaluation transforms are deterministic. Cache only images participating
    # in selected pairs so every radius reuses exactly the same tensor instead
    # of repeatedly decoding the image from disk.
    image_cache = _cache_collision_images(dataset, pairs)
    shard_dir = evaluation_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_frames = []
    epsilons = sorted({0.0, *(float(value) for value in config["attack"]["input_epsilons"])})
    for epsilon_index, epsilon in enumerate(epsilons):
        for start in range(0, len(pairs), collision_batch_size):
            batch_pairs = pairs[start : start + collision_batch_size]
            attack_seed = (
                int(config["attack"].get("seed", 2025)) + round(epsilon * 255) * 100_000 + start
            )
            shard_path = shard_dir / f"epsilon-{epsilon_index:02d}-batch-{start:06d}.parquet"
            if shard_path.exists():
                shard = pd.read_parquet(shard_path)
                expected_sources = [sample_ids[pair[0]] for pair in batch_pairs]
                expected_targets = [sample_ids[pair[1]] for pair in batch_pairs]
                valid = (
                    len(shard) == len(batch_pairs)
                    and shard.source_sample_id.astype(str).tolist() == expected_sources
                    and shard.target_sample_id.astype(str).tolist() == expected_targets
                    and set(shard.epsilon.astype(float)) == {epsilon}
                    and set(shard.attack_seed.astype(int)) == {attack_seed}
                    and set(shard.attack_protocol_version.astype(int)) == {ATTACK_PROTOCOL_VERSION}
                )
                if not valid:
                    raise ValueError(f"Collision resume shard does not match: {shard_path}")
                shard_frames.append(shard)
                continue
            source_indices = [pair[0] for pair in batch_pairs]
            target_indices = [pair[1] for pair in batch_pairs]
            source_images = torch.stack([image_cache[index] for index in source_indices]).to(
                device, non_blocking=True
            )
            target_images = torch.stack([image_cache[index] for index in target_indices]).to(
                device, non_blocking=True
            )
            source_labels = labels[source_indices].to(device, non_blocking=True)
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
                seed=attack_seed,
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
                linf_norm = (result["adversarial"] - source_images).abs().flatten(1).amax(1)
            distances_cpu = distances.detach().cpu()
            distance_percentiles = (
                torch.searchsorted(
                    sorted_reference_distances,
                    distances_cpu.double(),
                    right=True,
                ).double()
                / len(reference_distances)
                * 100.0
            )
            source_prediction_cpu = source_prediction.cpu()
            target_prediction_cpu = target_class_prediction.cpu()
            success_cpu = result["successful"].cpu()
            linf_cpu = linf_norm.cpu()
            final_codes = final_output.metadata.get("code_indices")
            target_codes = target_output.metadata.get("code_indices")
            if final_codes is not None and target_codes is not None:
                code_equal = (final_codes == target_codes).flatten(1).cpu()
            else:
                code_equal = None
            final_quantized = final_output.metadata.get("quantized_latent")
            target_quantized = target_output.metadata.get("quantized_latent")
            if final_quantized is not None and target_quantized is not None:
                quantized_equal = (final_quantized == target_quantized).flatten(1).cpu()
            else:
                quantized_equal = None
            batch_rows = []
            for local, (source_index, target_index) in enumerate(batch_pairs):
                row = {
                    "source_sample_id": sample_ids[source_index],
                    "target_sample_id": sample_ids[target_index],
                    "source_label": int(labels[source_index]),
                    "target_label": int(labels[target_index]),
                    "epsilon": epsilon,
                    "distance": float(distances_cpu[local]),
                    "distance_percentile": float(distance_percentiles[local]),
                    "reference_source_preserved": bool(
                        source_prediction_cpu[local] == labels[source_index]
                    ),
                    "target_class_prediction": int(target_prediction_cpu[local]),
                    "target_class_reached": bool(
                        target_prediction_cpu[local] == labels[target_index]
                    ),
                    "collision_success": bool(success_cpu[local]),
                    "collision_criterion": result["criterion"],
                    "linf_norm": float(linf_cpu[local]),
                    "input_bound_satisfied": bool(linf_cpu[local] <= epsilon + 1e-6),
                    "lambda_sem": lambda_sem,
                    "attack_seed": attack_seed,
                    "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
                }
                if result["criterion"] == "threshold":
                    row["continuous_collision"] = bool(distances_cpu[local] < threshold)
                if code_equal is not None:
                    row["exact_vq_collision"] = bool(code_equal[local].all())
                    row["vq_token_match_fraction"] = float(code_equal[local].float().mean())
                if quantized_equal is not None:
                    row["exact_quantized_collision"] = bool(quantized_equal[local].all())
                    row["quantized_bin_match_fraction"] = float(
                        quantized_equal[local].float().mean()
                    )
                batch_rows.append(row)
            shard = pd.DataFrame(batch_rows)
            save_frame(shard, shard_path)
            shard_frames.append(shard)
    path = evaluation_dir / "collision_attacks.parquet"
    save_frame(pd.concat(shard_frames, ignore_index=True), path)
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
    write_json(
        evaluation_dir / "config.json", {**evaluation_config, "environment": environment_info()}
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
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
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
        images = torch.stack([sample["image"] for sample in samples]).to(device, non_blocking=True)
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
                linf_norm = (result.adversarial - images).abs().flatten(1).amax(1)
            labels_cpu = labels.cpu()
            clean_prediction_cpu = clean_logits.argmax(1).cpu()
            adversarial_prediction_cpu = adversarial_logits.argmax(1).cpu()
            result_loss_cpu = result.loss.cpu()
            result_retained_cpu = result.retained_loss.cpu()
            result_initial_cpu = result.initial_loss.cpu()
            linf_cpu = linf_norm.cpu()
            for index, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels_cpu[index]),
                        "radius": epsilon,
                        "clean_prediction": int(clean_prediction_cpu[index]),
                        "clean_correct": bool(clean_prediction_cpu[index] == labels_cpu[index]),
                        "adversarial_prediction": int(adversarial_prediction_cpu[index]),
                        "successful": bool(adversarial_prediction_cpu[index] != labels_cpu[index]),
                        "loss": float(result_loss_cpu[index]),
                        "retained_loss": float(result_retained_cpu[index]),
                        "initial_loss": float(result_initial_cpu[index]),
                        "queries": int(config["attack"].get("square_queries", 5000)),
                        "attack_seed": radius_seed,
                        "checkpoint": checkpoint,
                        "linf_norm": float(linf_cpu[index]),
                        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
                    }
                )
    path = evaluation_dir / "square_attack.parquet"
    save_frame(pd.DataFrame(rows), path)
    write_json(
        path.with_suffix(".json"),
        {**evaluation_config, "environment": environment_info()},
    )
    write_json(
        evaluation_dir / "config.json", {**evaluation_config, "environment": environment_info()}
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path


def _final_geometry_effective_rank(run_dir: Path) -> float:
    matches = []
    for candidate in (run_dir / "analysis").glob("geometry-*-attempt*"):
        config_path = candidate / "config.json"
        metric_path = candidate / "geometry.json"
        if (
            not (candidate / "COMPLETED").exists()
            or not config_path.exists()
            or not metric_path.exists()
        ):
            continue
        analysis_config = json.loads(config_path.read_text(encoding="utf-8"))
        if analysis_config.get("split") == "final" and analysis_config.get("max_samples") is None:
            matches.append(metric_path)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one complete final geometry analysis for {run_dir}, found {matches}"
        )
    metrics = json.loads(matches[0].read_text(encoding="utf-8"))
    if "effective_rank" not in metrics:
        raise ValueError(f"Geometry analysis lacks effective_rank: {matches[0]}")
    return float(metrics["effective_rank"])


def validate_nearest_capacity_source(
    target_run_dir: str | Path,
    source_run_dir: str | Path | None,
    target_config: dict[str, Any],
) -> dict[str, Any]:
    """Verify that transfer uses the nearest measured-capacity continuous run."""
    target_run_dir = Path(target_run_dir)
    proposed_source = Path(source_run_dir).resolve() if source_run_dir is not None else None
    target_family = str(target_config["model"].get("family", "")).lower()
    if target_family not in {"vq", "quantized", "quantized_continuous"}:
        raise ValueError("The registered transfer diagnostic applies only to discrete models")
    target_rank = _final_geometry_effective_rank(target_run_dir)
    target_seed = int(target_config.get("seed", -1))
    target_manifest_path = target_run_dir / "data_manifest_hash.txt"
    if not target_manifest_path.exists():
        raise ValueError(f"Missing target data-manifest hash: {target_manifest_path}")
    target_manifest = target_manifest_path.read_text(encoding="utf-8").strip()
    candidates: list[tuple[Path, float]] = []
    missing_geometry = []
    output_root = target_run_dir.parent.parent
    for family in ("dimensional", "vib"):
        for candidate in (output_root / family).glob("*"):
            config_path = candidate / "resolved_config.yaml"
            hash_path = candidate / "data_manifest_hash.txt"
            if not (candidate / "COMPLETED").exists() or not config_path.exists():
                continue
            candidate_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if int(candidate_config.get("seed", -2)) != target_seed:
                continue
            if (
                not hash_path.exists()
                or hash_path.read_text(encoding="utf-8").strip() != target_manifest
            ):
                continue
            try:
                rank = _final_geometry_effective_rank(candidate)
            except ValueError:
                missing_geometry.append(str(candidate))
                continue
            candidates.append((candidate, rank))
    if missing_geometry:
        raise ValueError(
            "Cannot establish nearest capacity because continuous candidates lack one complete "
            f"final geometry analysis: {missing_geometry}"
        )
    if not candidates:
        raise ValueError("No seed- and manifest-matched continuous capacity candidates found")
    minimum_gap = min(abs(rank - target_rank) for _, rank in candidates)
    nearest = [
        path.resolve()
        for path, rank in candidates
        if abs(abs(rank - target_rank) - minimum_gap) <= 1e-12
    ]
    nearest = sorted(nearest, key=str)
    if proposed_source is not None and proposed_source not in nearest:
        raise ValueError(
            "Transfer source is not the nearest measured-capacity continuous model; "
            f"eligible nearest runs are {[str(path) for path in nearest]}"
        )
    selected_source = proposed_source or nearest[0]
    source_rank = next(rank for path, rank in candidates if path.resolve() == selected_source)
    return {
        "matched_source_run": str(selected_source),
        "capacity_metric": "effective_rank",
        "target_effective_rank": target_rank,
        "source_effective_rank": source_rank,
        "absolute_capacity_gap": abs(source_rank - target_rank),
        "matched_seed": target_seed,
    }


def evaluate_transfer_attack(
    target_run_dir: str | Path,
    source_run_dir: str | Path | None = None,
    split: str = "final",
    max_samples: int = 1000,
) -> Path:
    """Attack a continuous source checkpoint and evaluate transfer to a target checkpoint."""
    target_run_dir = Path(target_run_dir)
    target_checkpoint = _resolve_checkpoint(target_run_dir, None)
    target, target_config, target_device = load_model(target_run_dir, target_checkpoint)
    capacity_match = validate_nearest_capacity_source(target_run_dir, source_run_dir, target_config)
    source_run_dir = Path(capacity_match["matched_source_run"])
    source_checkpoint = _resolve_checkpoint(source_run_dir, None)
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
        **capacity_match,
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
    }
    evaluation_dir = _new_evaluation_dir(target_run_dir, "transfer", evaluation_config)
    rows = []
    batch_size = int(target_config["attack"].get("evaluation_batch_size", 32))
    base_seed = int(target_config["attack"].get("seed", 2025))
    for start in range(0, len(selected_indices), batch_size):
        indices = selected_indices[start : start + batch_size]
        samples = [dataset[index] for index in indices]
        images = torch.stack([sample["image"] for sample in samples]).to(
            target_device, non_blocking=True
        )
        labels = torch.tensor([sample["label"] for sample in samples], device=target_device)
        with torch.no_grad():
            target_clean = target(images, sample=False).logits.argmax(1)
        previous_source_candidate = images
        for epsilon in sorted(
            {float(value) for value in target_config["attack"]["input_epsilons"]}
        ):
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
                previous_source_candidate,
            )
            previous_source_candidate = result.adversarial.detach()
            with torch.no_grad():
                target_adversarial = target(result.adversarial, sample=False).logits.argmax(1)
                linf_norm = (result.adversarial - images).abs().flatten(1).amax(1)
            labels_cpu = labels.cpu()
            target_clean_cpu = target_clean.cpu()
            target_adversarial_cpu = target_adversarial.cpu()
            linf_cpu = linf_norm.cpu()
            restart_cpu = result.restart.cpu() if result.restart is not None else None
            for local, sample in enumerate(samples):
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "label": int(labels_cpu[local]),
                        "radius": epsilon,
                        "clean_prediction": int(target_clean_cpu[local]),
                        "clean_correct": bool(target_clean_cpu[local] == labels_cpu[local]),
                        "adversarial_prediction": int(target_adversarial_cpu[local]),
                        "successful": bool(target_adversarial_cpu[local] != labels_cpu[local]),
                        "linf_norm": float(linf_cpu[local]),
                        "best_restart": (
                            int(restart_cpu[local]) if restart_cpu is not None else -1
                        ),
                        "attack_seed": radius_seed,
                        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
                    }
                )
    path = evaluation_dir / "transfer_attack.parquet"
    save_frame(pd.DataFrame(rows), path)
    write_json(
        evaluation_dir / "config.json", {**evaluation_config, "environment": environment_info()}
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return path
