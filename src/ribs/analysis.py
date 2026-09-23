"""Analysis, aggregation, and statistical reporting from saved artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from .artifacts import load_latents, save_frame
from .collisions import (
    calibrate_collision_threshold,
    collision_records,
    natural_collision_metrics,
    nearest_opposing,
)
from .config import config_hash
from .data import manifest_hash, sample_ids_hash, sha256_file
from .evaluation import (
    ATTACK_PROTOCOL_VERSION,
    _ReconstructionTask,
    _resolve_checkpoint,
    _stratified_dataset_indices,
    extract_latents,
    load_model,
    make_loader,
)
from .geometry import contraction_metrics, encoder_spectral_norm, geometry_metrics
from .invariance import invariance_metrics
from .metrics import shared_clean_correct, summarize_curve
from .phase2 import (
    PHASE2_SPECS,
    canonical_completed_runs,
    decision_record_hash,
    phase2_strength_columns,
    validate_decision_record,
)
from .utils import environment_info, write_json

ANALYSIS_PROTOCOL_VERSION = 2

CONFIGURATION_COLUMNS = (
    "family",
    "dz",
    "beta",
    "codebook_size",
    "bits",
    "strength_parameter",
    "strength_value",
    "bottleneck_strength_ordinal",
)


def _configuration_columns(frame: pd.DataFrame, *extra: str) -> list[str]:
    """Return every available field that uniquely identifies a configuration."""
    return [column for column in (*CONFIGURATION_COLUMNS, *extra) if column in frame]


def _new_analysis_dir(run_dir: Path, kind: str, analysis_config: dict[str, Any]) -> Path:
    base_name = f"{kind}-{config_hash(analysis_config)}"
    attempt = 0
    while True:
        candidate = run_dir / "analysis" / f"{base_name}-attempt{attempt}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            attempt += 1


def _completed_analysis_file(
    run_dir: Path,
    kind: str,
    split: str,
    filename: str,
    legacy_filename: str | None = None,
    *,
    allow_legacy: bool = False,
) -> Path | None:
    matches = []
    for candidate in (run_dir / "analysis").glob(f"{kind}-*-attempt*"):
        config_path = candidate / "config.json"
        if not (candidate / "COMPLETED").exists() or not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("split") == split and config.get("max_samples") is None:
            path = candidate / filename
            if path.exists():
                matches.append(path)
    if len(matches) > 1:
        raise ValueError(f"Multiple complete {kind} analyses found for {run_dir}: {matches}")
    if matches:
        return matches[0]
    if allow_legacy:
        legacy = run_dir / "analysis" / (legacy_filename or filename)
        return legacy if legacy.exists() else None
    return None


def _stratified_sample_indices(index: pd.DataFrame, count: int) -> list[int]:
    """Choose stable, class-balanced sample IDs shared across checkpoints."""
    pools: dict[int, list[int]] = {}
    for class_label, group in index.groupby("label", sort=True):
        pools[int(class_label)] = sorted(
            group.index.tolist(),
            key=lambda row: hashlib.sha256(str(index.loc[row, "sample_id"]).encode()).digest(),
        )
    selected: list[int] = []
    offset = 0
    while len(selected) < min(count, len(index)):
        added = False
        for class_label in sorted(pools):
            if offset < len(pools[class_label]) and len(selected) < count:
                selected.append(pools[class_label][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def analyze_geometry(
    run_dir: str | Path,
    split: str = "final",
    jacobian_samples: int = 500,
    max_samples: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, None)
    analysis_config = {
        "kind": "geometry",
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "split": split,
        "max_samples": max_samples,
        "jacobian_samples": jacobian_samples,
    }
    extract_latents(run_dir, split, checkpoint)
    analysis_dir = _new_analysis_dir(run_dir, "geometry", analysis_config)
    write_json(analysis_dir / "config.json", {**analysis_config, "environment": environment_info()})
    latent_path = run_dir / "artifacts" / f"latents_{split}.safetensors"
    index_path = run_dir / "artifacts" / f"latents_{split}_index.parquet"
    tensors = load_latents(latent_path)
    index = pd.read_parquet(index_path)
    if max_samples is not None:
        tensors = {key: value[:max_samples] for key, value in tensors.items()}
        index = index.iloc[:max_samples].reset_index(drop=True)
    labels = torch.as_tensor(index.label.to_numpy())
    metrics = geometry_metrics(tensors["canonical_latent"], labels)
    model, config, device = load_model(run_dir, checkpoint)
    loader = make_loader(
        config, split, batch_size=int(config["attack"].get("evaluation_batch_size", 32))
    )
    image_size = int(config["data"].get("image_size", 224))
    raw_path = analysis_dir / ".raw_images.dat"
    raw_images = np.memmap(
        raw_path, dtype="float32", mode="w+", shape=(len(index), 3, image_size, image_size)
    )
    image_ids = []
    collected = 0
    for batch in loader:
        take = (
            len(batch["label"])
            if max_samples is None
            else min(len(batch["label"]), max_samples - collected)
        )
        if take <= 0:
            break
        raw_images[collected : collected + take] = batch["image"][:take].numpy()
        image_ids.extend(batch["sample_id"][:take])
        collected += take
    raw_images.flush()
    image_tensor = torch.from_numpy(raw_images)
    metrics.update(
        contraction_metrics(
            image_tensor.cpu(), tensors["canonical_latent"], labels, image_ids[: len(index)]
        )
    )
    jacobian_indices = _stratified_sample_indices(index, jacobian_samples)
    jacobian_images = image_tensor[jacobian_indices].to(device)
    jacobian = encoder_spectral_norm(model, jacobian_images, seed=2025)
    if len(jacobian):
        metrics["encoder_jacobian_median"] = float(jacobian.median())
        metrics["encoder_jacobian_p95"] = float(torch.quantile(jacobian, 0.95))
    del image_tensor, raw_images
    raw_path.unlink(missing_ok=True)
    write_json(analysis_dir / "geometry.json", metrics)
    distances, indices = nearest_opposing(tensors["canonical_latent"], labels)
    collision = natural_collision_metrics(tensors["canonical_latent"], labels)
    tuning_path = run_dir / "artifacts" / "latents_development_tune.safetensors"
    tuning_index_path = run_dir / "artifacts" / "latents_development_tune_index.parquet"
    extract_latents(run_dir, "development_tune", checkpoint)
    if tuning_path.exists() and tuning_index_path.exists():
        tuning = load_latents(tuning_path)
        tuning_index = pd.read_parquet(tuning_index_path)
        tuning_labels = torch.as_tensor(tuning_index.label.to_numpy())
        thresholds = {
            percentile: calibrate_collision_threshold(
                tuning["canonical_latent"], tuning_labels, percentile
            )
            for percentile in (1.0, 5.0, 10.0)
        }
        threshold = thresholds[5.0]
        collision.update(natural_collision_metrics(tensors["canonical_latent"], labels, threshold))
        collision.update(
            {
                f"collision_rate_tau_p{int(percentile)}": float((distances < value).double().mean())
                for percentile, value in thresholds.items()
            }
        )
        collision.update(
            {f"collision_threshold_p{int(key)}": value for key, value in thresholds.items()}
        )
    else:
        threshold = float("nan")
        collision["collision_threshold"] = threshold
    collision_frame = collision_records(
        index.sample_id.tolist(), labels, distances, indices, threshold
    )
    codes_path = run_dir / "artifacts" / f"codes_{split}.safetensors"
    if codes_path.exists():
        codes = load_latents(codes_path)["code_indices"]
        opposing_codes = codes[indices]
        matches = codes == opposing_codes
        raw_codebook_distance = (
            (tensors["canonical_latent"] - tensors["canonical_latent"][indices])
            .flatten(1)
            .norm(dim=1)
        )
        collision_frame["vq_exact_sequence_match"] = matches.all(dim=1).tolist()
        collision_frame["vq_token_match_fraction"] = matches.double().mean(dim=1).tolist()
        collision_frame["vq_hamming_distance"] = (~matches).sum(dim=1).tolist()
        collision_frame["vq_codebook_euclidean_distance"] = raw_codebook_distance.tolist()
        collision["vq_exact_sequence_collision_rate"] = float(matches.all(dim=1).double().mean())
        collision["vq_mean_token_match_fraction"] = float(matches.double().mean())
        collision["vq_median_hamming_distance"] = float((~matches).sum(dim=1).median())
        collision["vq_median_codebook_euclidean_distance"] = float(raw_codebook_distance.median())
    save_frame(collision_frame, analysis_dir / "natural_collisions.parquet")
    write_json(analysis_dir / "natural_collision_summary.json", collision)
    (analysis_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return {**metrics, **collision}


def summarize_attack_file(path: str | Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    curve, summary = summarize_curve(frame)
    summary["attack_file"] = str(path)
    save_frame(curve, Path(path).with_suffix(".summary.parquet"))
    write_json(Path(path).with_suffix(".summary.json"), summary)
    return summary


def analyze_invariance(
    run_dir: str | Path,
    split: str = "final",
    max_samples: int | None = None,
    reference_run_dir: str | Path | None = None,
) -> dict[str, float]:
    run_dir = Path(run_dir)
    checkpoint = _resolve_checkpoint(run_dir, None)
    model, config, device = load_model(run_dir, checkpoint)
    analysis_config: dict[str, Any] = {
        "kind": "invariance",
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / checkpoint),
        "split": split,
        "max_samples": max_samples,
    }
    if config["model"].get("family") == "autoencoder":
        if reference_run_dir is None:
            raise ValueError("Autoencoder invariance requires reference_run_dir")
        reference, reference_config, reference_device = load_model(reference_run_dir)
        if reference_device != device:
            raise ValueError("Autoencoder and reference classifier must use the same device")
        if manifest_hash(config["data"]["manifest"]) != manifest_hash(
            reference_config["data"]["manifest"]
        ):
            raise ValueError("Autoencoder and reference classifier use different data manifests")
        if str(reference_config.get("model", {}).get("family", "")).lower() != "identity":
            raise ValueError("Autoencoder invariance requires an identity reference classifier")
        if int(reference_config.get("seed", -1)) != 0:
            raise ValueError("Autoencoder invariance requires the seed-0 reference classifier")
        reference_checkpoint = _resolve_checkpoint(Path(reference_run_dir), None)
        analysis_config.update(
            {
                "reference_checkpoint": str(
                    Path(reference_run_dir) / "checkpoints" / reference_checkpoint
                ),
                "reference_checkpoint_sha256": sha256_file(
                    Path(reference_run_dir) / "checkpoints" / reference_checkpoint
                ),
            }
        )
        model = _ReconstructionTask(model, reference).to(device).eval()
    extract_latents(run_dir, split, checkpoint)
    analysis_dir = _new_analysis_dir(run_dir, "invariance", analysis_config)
    write_json(analysis_dir / "config.json", {**analysis_config, "environment": environment_info()})
    loader = make_loader(
        config, split, batch_size=int(config["attack"].get("evaluation_batch_size", 32))
    )
    batches: list[dict[str, float]] = []
    batch_sizes: list[int] = []
    seen = 0
    for batch in loader:
        if max_samples is not None and seen >= max_samples:
            break
        take = (
            len(batch["label"])
            if max_samples is None
            else min(len(batch["label"]), max_samples - seen)
        )
        images = batch["image"][:take].to(device)
        labels = torch.as_tensor(batch["label"][:take], device=device)
        batches.append(invariance_metrics(model, images, labels, list(batch["sample_id"][:take])))
        batch_sizes.append(take)
        seen += take
    keys = batches[0].keys() if batches else []
    result = {
        key: float(np.average([batch[key] for batch in batches], weights=batch_sizes))
        for key in keys
    }
    latent_path = run_dir / "artifacts" / f"latents_{split}.safetensors"
    index_path = run_dir / "artifacts" / f"latents_{split}_index.parquet"
    latent_tensors = load_latents(latent_path)
    latent_index = pd.read_parquet(index_path)
    if max_samples is not None:
        latent_tensors = {key: value[:max_samples] for key, value in latent_tensors.items()}
        latent_index = latent_index.iloc[:max_samples]
    semantic = geometry_metrics(
        latent_tensors["canonical_latent"], torch.as_tensor(latent_index.label.to_numpy())
    )["inter_class_distance"]
    result["semantic_distance"] = float(semantic)
    result["semantic_to_nuisance_ratio"] = float(
        semantic / (result["nuisance_distance_macro"] + 1e-12)
    )
    write_json(analysis_dir / "invariance.json", result)
    (analysis_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return result


def aggregate_completed_runs(
    output_root: str | Path = "outputs",
    output_prefix: str = "phase1",
    include_families: list[str] | tuple[str, ...] | None = None,
    runtime_amendment: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate completed records into a namespaced result table.

    Phase 1 keeps its historical filenames by default. Phase 2 passes a
    separate prefix and family filter so new analysis never overwrites the
    validated Phase 1 outputs.
    """
    rows = []
    curve_rows = []
    sample_rows = []
    distance_rows = []
    collision_attack_rows = []
    representation_rows = []
    clean_records: dict[str, pd.DataFrame] = {}
    allowed = set(include_families) if include_families is not None else None
    completed_runs = []
    for path in Path(output_root).glob("*/*"):
        if not (path / "COMPLETED").exists():
            continue
        config_path = path / "resolved_config.yaml"
        family = None
        if config_path.exists():
            family = (
                (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
                .get("model", {})
                .get("family")
            )
        if allowed is None or family in allowed:
            completed_runs.append(path)
    phase2_families = {"vib", "vq", "quantized", "autoencoder"}
    amendment = None
    if output_prefix == "phase2" and (allowed is None or allowed & phase2_families):
        from .phase2 import canonical_completed_runs

        families = tuple(sorted((allowed or phase2_families) & phase2_families))
        canonical, _ = canonical_completed_runs(output_root, families)
        canonical_set = {str(path) for path in canonical}
        completed_runs = [
            path
            for path in completed_runs
            if path.parent.name not in phase2_families or str(path) in canonical_set
        ]
        if runtime_amendment is not None:
            from .phase2 import load_runtime_amendment

            amendment = load_runtime_amendment(runtime_amendment)
    manifest_hashes = {
        (run_dir / "data_manifest_hash.txt").read_text().strip()
        for run_dir in completed_runs
        if (run_dir / "data_manifest_hash.txt").exists()
    }
    if len(manifest_hashes) > 1:
        raise ValueError("Completed runs use different data-manifest hashes")
    for run_dir in completed_runs:
        if not (run_dir / "COMPLETED").exists():
            continue
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_metrics = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        base = {
            "run_dir": str(run_dir),
            "seed": config.get("seed"),
            "codebook_collapsed": bool(run_metrics.get("codebook_collapsed", False)),
            **config.get("model", {}),
        }
        include_robustness = True
        include_collision = True
        if amendment is not None:
            from .phase2 import runtime_policy_for_config

            runtime_policy = runtime_policy_for_config(config, amendment)
            include_robustness = (
                base.get("family") == "dimensional"
                or runtime_policy["phase2_robustness"]
            )
            include_collision = runtime_policy["phase3_collision"]
        if base.get("family") == "autoencoder":
            clean_summaries = [
                candidate / "reconstruction_final.json"
                for candidate in (run_dir / "evaluations").glob("autoencoder-clean-*-attempt*")
                if (candidate / "COMPLETED").exists()
                and (candidate / "reconstruction_final.json").exists()
            ]
            if len(clean_summaries) > 1:
                raise ValueError(f"Multiple final autoencoder clean evaluations for {run_dir}")
            if clean_summaries:
                clean_summary = json.loads(clean_summaries[0].read_text(encoding="utf-8"))
                for key in (
                    "reference_original_accuracy",
                    "reconstruction_accuracy",
                    "reconstruction_mse",
                    "reconstruction_psnr_db",
                ):
                    if key in clean_summary:
                        base[key] = clean_summary[key]
        else:
            clean_summaries = []
            for candidate in (run_dir / "evaluations").glob("clean-*-attempt*"):
                summary_path = candidate / "clean_final.json"
                if not (candidate / "COMPLETED").exists() or not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("split") == "final" and summary.get("max_samples") is None:
                    clean_summaries.append(summary_path)
            if len(clean_summaries) > 1:
                raise ValueError(f"Multiple final clean evaluations for {run_dir}")
            if clean_summaries:
                clean_summary = json.loads(clean_summaries[0].read_text(encoding="utf-8"))
                if "clean_accuracy" in clean_summary:
                    base["clean_accuracy"] = clean_summary["clean_accuracy"]
        evaluation_dirs = []
        candidates = (
            [
                *(run_dir / "evaluations").glob("robustness-*-attempt*"),
                *(run_dir / "evaluations").glob("autoencoder-robustness-*-attempt*"),
            ]
            if include_robustness
            else []
        )
        for candidate in candidates:
            config_file = candidate / "config.json"
            if not (candidate / "COMPLETED").exists() or not config_file.exists():
                continue
            evaluation_config = json.loads(config_file.read_text())
            if (
                evaluation_config.get("split") == "final"
                and evaluation_config.get("max_samples") is None
                and evaluation_config.get("attack_protocol_version") == ATTACK_PROTOCOL_VERSION
            ):
                evaluation_dirs.append(candidate)
        if len(evaluation_dirs) > 1:
            raise ValueError(f"Multiple final robustness evaluations found for {run_dir}")
        evaluation_dir = evaluation_dirs[0] if evaluation_dirs else run_dir / "evaluations"
        input_records: pd.DataFrame | None = None
        attack_files = {
            "input_pgd": ("input_pgd.parquet", "input_pgd_reference.parquet"),
            "latent_pgd": ("latent_pgd.parquet", "latent_pgd_reference.parquet"),
        }
        for attack_name, filenames in attack_files.items():
            path = next(
                (
                    evaluation_dir / filename
                    for filename in filenames
                    if (evaluation_dir / filename).exists()
                ),
                evaluation_dir / filenames[0],
            )
            if path.exists():
                records = pd.read_parquet(path).sort_values(["sample_id", "radius"])
                records["nested_success"] = records.groupby("sample_id")["successful"].cummax()
                records["robust"] = (~records["nested_success"]).astype(float)
                sample_rows.extend(
                    {**base, "attack": attack_name, **record}
                    for record in records[["sample_id", "radius", "robust"]].to_dict("records")
                )
                curve, summary = summarize_curve(records)
                if attack_name == "input_pgd":
                    input_records = records
                    clean_records[str(run_dir)] = records[
                        ["sample_id", "clean_correct"]
                    ].drop_duplicates("sample_id")
                rows.append({**base, "attack": attack_name, **summary})
                curve_rows.extend(
                    [
                        {**base, "attack": attack_name, **record}
                        for record in curve.to_dict("records")
                    ]
                )
        if input_records is not None and config.get("model", {}).get("family") in {
            "vq",
            "quantized",
            "quantized_continuous",
        }:
            square_dirs = [
                candidate
                for candidate in (run_dir / "evaluations").glob("square-*-attempt*")
                if (candidate / "COMPLETED").exists()
                and (candidate / "square_attack.parquet").exists()
                and (candidate / "config.json").exists()
                and json.loads((candidate / "config.json").read_text()).get("split") == "final"
                and json.loads((candidate / "config.json").read_text()).get("max_samples")
                == 1000
                and json.loads((candidate / "config.json").read_text()).get(
                    "attack_protocol_version"
                )
                == ATTACK_PROTOCOL_VERSION
            ]
            if len(square_dirs) > 1:
                raise ValueError(f"Multiple completed Square Attack evaluations for {run_dir}")
            if square_dirs:
                square = pd.read_parquet(square_dirs[0] / "square_attack.parquet").sort_values(
                    ["sample_id", "radius"]
                )
                square["nested_success"] = square.groupby("sample_id")["successful"].cummax()
                square["robust"] = (~square["nested_success"]).astype(float)
                square_curve, square_summary = summarize_curve(square)
                rows.append({**base, "attack": "input_square", **square_summary})
                curve_rows.extend(
                    {**base, "attack": "input_square", **record}
                    for record in square_curve.to_dict("records")
                )
                sample_rows.extend(
                    {**base, "attack": "input_square", **record}
                    for record in square[["sample_id", "radius", "robust"]].to_dict("records")
                )
                pgd_subset = input_records.merge(
                    square[["sample_id", "radius"]].drop_duplicates(),
                    on=["sample_id", "radius"],
                    how="inner",
                )
                combined = pgd_subset.merge(
                    square[["sample_id", "radius", "successful"]],
                    on=["sample_id", "radius"],
                    suffixes=("_pgd", "_square"),
                )
                combined["successful"] = combined.successful_pgd.astype(
                    bool
                ) | combined.successful_square.astype(bool)
                combined = combined.sort_values(["sample_id", "radius"])
                combined["nested_success"] = combined.groupby("sample_id")["successful"].cummax()
                combined["robust"] = (~combined["nested_success"]).astype(float)
                sample_rows.extend(
                    {**base, "attack": "input_strongest", **record}
                    for record in combined[["sample_id", "radius", "robust"]].to_dict("records")
                )
                combined_curve, combined_summary = summarize_curve(combined)
                rows.append({**base, "attack": "input_strongest", **combined_summary})
                curve_rows.extend(
                    {**base, "attack": "input_strongest", **record}
                    for record in combined_curve.to_dict("records")
                )
            transfer_dirs = [
                candidate
                for candidate in (run_dir / "evaluations").glob("transfer-*-attempt*")
                if (candidate / "COMPLETED").exists()
                and (candidate / "transfer_attack.parquet").exists()
                and (candidate / "config.json").exists()
                and json.loads((candidate / "config.json").read_text()).get("split") == "final"
                and json.loads((candidate / "config.json").read_text()).get("max_samples")
                == 1000
                and json.loads((candidate / "config.json").read_text()).get(
                    "attack_protocol_version"
                )
                == ATTACK_PROTOCOL_VERSION
            ]
            if len(transfer_dirs) > 1:
                raise ValueError(f"Multiple completed transfer evaluations for {run_dir}")
            if transfer_dirs:
                transfer = pd.read_parquet(
                    transfer_dirs[0] / "transfer_attack.parquet"
                ).sort_values(["sample_id", "radius"])
                transfer["nested_success"] = transfer.groupby("sample_id")["successful"].cummax()
                transfer["robust"] = (~transfer["nested_success"]).astype(float)
                transfer_curve, transfer_summary = summarize_curve(transfer)
                rows.append({**base, "attack": "input_transfer", **transfer_summary})
                curve_rows.extend(
                    {**base, "attack": "input_transfer", **record}
                    for record in transfer_curve.to_dict("records")
                )
                sample_rows.extend(
                    {**base, "attack": "input_transfer", **record}
                    for record in transfer[["sample_id", "radius", "robust"]].to_dict("records")
                )
        geometry_path = _completed_analysis_file(
            run_dir, "geometry", "final", "geometry.json", "geometry_final.json"
        )
        if geometry_path is not None:
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            base.update(geometry)
            for row in rows:
                if row["run_dir"] == str(run_dir):
                    row.update(geometry)
        invariance_path = _completed_analysis_file(
            run_dir, "invariance", "final", "invariance.json", "invariance_final.json"
        )
        if invariance_path is not None:
            invariance = json.loads(invariance_path.read_text(encoding="utf-8"))
            base.update(invariance)
            for row in rows:
                if row["run_dir"] == str(run_dir):
                    row.update(invariance)
        collision_path = _completed_analysis_file(
            run_dir,
            "geometry",
            "final",
            "natural_collisions.parquet",
            "natural_collisions_final.parquet",
        )
        if collision_path is not None:
            natural_records = pd.read_parquet(collision_path)
            distance_rows.extend(
                {**base, **record}
                for record in natural_records[["sample_id", "distance"]].to_dict("records")
            )
        natural_summary_path = _completed_analysis_file(
            run_dir,
            "geometry",
            "final",
            "natural_collision_summary.json",
        )
        if natural_summary_path is not None:
            base.update(json.loads(natural_summary_path.read_text(encoding="utf-8")))
        representation_rows.append(base.copy())
        collision_dirs = []
        collision_candidates = (
            (run_dir / "evaluations").glob("collision-*-attempt*")
            if include_collision
            else ()
        )
        for candidate in collision_candidates:
            if (
                not (candidate / "COMPLETED").exists()
                or not (candidate / "collision_attacks.parquet").exists()
            ):
                continue
            metadata_path = candidate / "config.json"
            if not metadata_path.exists():
                metadata_path = candidate / "collision_attacks.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else {}
            )
            if (
                metadata.get("split", "final") == "final"
                and metadata.get("attack_protocol_version") == ATTACK_PROTOCOL_VERSION
            ):
                collision_dirs.append(candidate)
        if len(collision_dirs) > 1:
            raise ValueError(f"Multiple completed collision evaluations found for {run_dir}")
        if collision_dirs:
            collision_attacks = pd.read_parquet(collision_dirs[0] / "collision_attacks.parquet")
            collision_attack_rows.extend(
                {**base, **record} for record in collision_attacks.to_dict("records")
            )
    frame = pd.DataFrame(rows)
    representation_frame = pd.DataFrame(representation_rows)
    output_path = Path(output_root)
    if output_prefix == "phase2" and not frame.empty:
        frame = phase2_strength_columns(frame)
        curve_rows = phase2_strength_columns(pd.DataFrame(curve_rows)).to_dict("records")
        sample_rows = phase2_strength_columns(pd.DataFrame(sample_rows)).to_dict("records")
    if output_prefix == "phase2" and not representation_frame.empty:
        representation_frame = phase2_strength_columns(representation_frame)
    save_frame(frame, output_path / f"{output_prefix}_summary.parquet")
    frame.to_csv(output_path / f"{output_prefix}_summary.csv", index=False)
    save_frame(pd.DataFrame(curve_rows), output_path / f"{output_prefix}_curves.parquet")
    if output_prefix == "phase2":
        save_frame(
            representation_frame,
            output_path / f"{output_prefix}_representation_summary.parquet",
        )
        representation_frame.to_csv(
            output_path / f"{output_prefix}_representation_summary.csv", index=False
        )
    shared_clean_frame = pd.DataFrame()
    if clean_records:
        shared_clean_frame = shared_clean_correct(clean_records)
        save_frame(
            shared_clean_frame,
            output_path / f"{output_prefix}_shared_clean_correct.parquet",
        )
    inferential_frame = frame[
        ~frame.get("codebook_collapsed", pd.Series(False, index=frame.index)).fillna(False)
    ].copy()
    excluded = frame.loc[
        frame.index.difference(inferential_frame.index),
        [column for column in ("run_dir", "family", "seed") if column in frame],
    ]
    write_json(
        output_path / f"{output_prefix}_inferential_exclusions.json",
        {
            "reason": "codebook collapse (<10% active codes for five consecutive epochs)",
            "runs": excluded.to_dict("records"),
        },
    )
    seed_level_summary(inferential_frame, output_root, output_prefix=output_prefix)
    primary_geometry = (
        inferential_frame[inferential_frame.family == "dimensional"]
        if "family" in inferential_frame
        else inferential_frame
    )
    if output_prefix == "phase2" and "family" in inferential_frame:
        correlation_parts = []
        for family, family_frame in inferential_frame.groupby("family", dropna=False):
            part = geometry_correlations(family_frame)
            if not part.empty:
                part.insert(0, "family", family)
                part.insert(0, "scope", "within_family")
                correlation_parts.append(part)
        cross_family = geometry_correlations(inferential_frame)
        if not cross_family.empty:
            cross_family.insert(0, "family", "all")
            cross_family.insert(0, "scope", "cross_family_exploratory")
            correlation_parts.append(cross_family)
        correlations = (
            pd.concat(correlation_parts, ignore_index=True) if correlation_parts else pd.DataFrame()
        )
    else:
        correlations = geometry_correlations(primary_geometry)
    save_frame(correlations, output_path / f"{output_prefix}_geometry_correlations.parquet")
    correlations.to_csv(output_path / f"{output_prefix}_geometry_correlations.csv", index=False)
    dimensional = inferential_frame[
        inferential_frame.get("family", pd.Series(index=inferential_frame.index)) == "dimensional"
    ].copy()
    if not dimensional.empty and {"dz", "seed", "attack", "robust_auc"}.issubset(dimensional):
        baseline = dimensional[dimensional.dz == 512][["seed", "attack", "robust_auc"]].rename(
            columns={"robust_auc": "robust_auc_dz512"}
        )
        contrasts = dimensional.merge(baseline, on=["seed", "attack"], how="inner")
        contrasts["paired_robust_auc_delta"] = contrasts.robust_auc - contrasts.robust_auc_dz512
        save_frame(contrasts, output_path / f"{output_prefix}_paired_contrasts.parquet")
        contrasts.to_csv(output_path / f"{output_prefix}_paired_contrasts.csv", index=False)
        mean_accuracy = dimensional.groupby(["attack", "dz"]).clean_accuracy.mean().reset_index()
        reference_accuracy = mean_accuracy[mean_accuracy.dz == 512][
            ["attack", "clean_accuracy"]
        ].rename(columns={"clean_accuracy": "reference_clean_accuracy"})
        matched = mean_accuracy.merge(reference_accuracy, on="attack")
        matched = matched[(matched.clean_accuracy - matched.reference_clean_accuracy).abs() <= 0.02]
        save_frame(matched, output_path / f"{output_prefix}_accuracy_matched_configs.parquet")
        dimensions = sorted(dimensional.dz.dropna().unique(), reverse=True)
        strength = {dimension: index for index, dimension in enumerate(dimensions)}
        dimensional["bottleneck_strength"] = dimensional.dz.map(strength)
        trends = {
            attack: ordinal_trends(group, "bottleneck_strength", "robust_auc")
            for attack, group in dimensional.groupby("attack")
        }
        distance_by_run = dimensional.drop_duplicates("run_dir")
        trends["median_nearest_opposing_distance"] = ordinal_trends(
            distance_by_run, "bottleneck_strength", "median_nearest_opposing_distance"
        )
        write_json(output_path / f"{output_prefix}_ordinal_trends.json", trends)
        matched_dimensions = matched[["attack", "dz"]].drop_duplicates()
        matched_rows = dimensional.merge(matched_dimensions, on=["attack", "dz"], how="inner")
        matched_trends = {
            attack: ordinal_trends(group, "bottleneck_strength", "robust_auc")
            for attack, group in matched_rows.groupby("attack")
        }
        write_json(
            output_path / f"{output_prefix}_accuracy_matched_trends.json",
            matched_trends,
        )
    identity = inferential_frame[
        inferential_frame.get("family", pd.Series(index=inferential_frame.index)) == "identity"
    ]
    if not identity.empty and not dimensional.empty:
        identity_comparison = identity.merge(
            dimensional[dimensional.dz == 512],
            on=["seed", "attack"],
            suffixes=("_identity", "_dimensional_512"),
        )
        if not identity_comparison.empty:
            identity_comparison["robust_auc_delta"] = (
                identity_comparison.robust_auc_identity
                - identity_comparison.robust_auc_dimensional_512
            )
        save_frame(
            identity_comparison,
            output_path / f"{output_prefix}_identity_diagnostic.parquet",
        )
    sample_frame = pd.DataFrame(sample_rows)
    if not sample_frame.empty and not shared_clean_frame.empty:
        shared_ids = set(shared_clean_frame.sample_id.astype(str))
        shared_samples = sample_frame[sample_frame.sample_id.astype(str).isin(shared_ids)].copy()
        shared_group_columns = _configuration_columns(shared_samples, "run_dir", "seed", "attack")
        shared_summaries = []
        for keys, group in shared_samples.groupby(shared_group_columns, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            records = group[["sample_id", "radius", "robust"]].copy()
            records["successful"] = records.robust.eq(0.0)
            records["clean_correct"] = True
            _, summary = summarize_curve(records)
            shared_summaries.append(
                {**dict(zip(shared_group_columns, key_values)), **summary, "scope": "shared_clean"}
            )
        save_frame(
            pd.DataFrame(shared_summaries),
            output_path / f"{output_prefix}_shared_clean_summary.parquet",
        )
    bootstrap_rows = []
    if not sample_frame.empty:
        sample_frame = sample_frame[~sample_frame["codebook_collapsed"].fillna(False).astype(bool)]
        group_columns = _configuration_columns(sample_frame, "attack", "radius")
        for keys, group in sample_frame.groupby(group_columns, dropna=False):
            estimate, lower, upper = hierarchical_bootstrap(group, "robust")
            key_values = keys if isinstance(keys, tuple) else (keys,)
            bootstrap_rows.append(
                {
                    **dict(zip(group_columns, key_values)),
                    "outcome": "robust_accuracy",
                    "point_estimate": float(group.robust.mean()),
                    "bootstrap_estimate": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    distance_frame = pd.DataFrame(distance_rows)
    if not distance_frame.empty:
        distance_frame = distance_frame[
            ~distance_frame["codebook_collapsed"].fillna(False).astype(bool)
        ]
        if output_prefix == "phase2":
            distance_frame = phase2_strength_columns(distance_frame)
        group_columns = _configuration_columns(distance_frame)
        for keys, group in distance_frame.groupby(group_columns, dropna=False):
            estimate, lower, upper = hierarchical_bootstrap(group, "distance", statistic="median")
            key_values = keys if isinstance(keys, tuple) else (keys,)
            bootstrap_rows.append(
                {
                    **dict(zip(group_columns, key_values)),
                    "outcome": "median_nearest_opposing_distance",
                    "point_estimate": float(group.distance.median()),
                    "bootstrap_estimate": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    save_frame(
        pd.DataFrame(bootstrap_rows),
        output_path / f"{output_prefix}_hierarchical_bootstrap.parquet",
    )
    collision_frame = pd.DataFrame(collision_attack_rows)
    collision_summary_frame = pd.DataFrame()
    if not collision_frame.empty:
        if output_prefix == "phase2":
            collision_frame = phase2_strength_columns(collision_frame)
        save_frame(
            collision_frame,
            output_path / f"{output_prefix}_collision_attack_records_all.parquet",
        )
        inferential_collisions = collision_frame[
            ~collision_frame["codebook_collapsed"].fillna(False).astype(bool)
        ].copy()
        pair_columns = ["source_sample_id", "target_sample_id"]
        configuration_column = "run_dir"
        pair_counts = inferential_collisions[
            pair_columns + [configuration_column]
        ].drop_duplicates()
        shared_pairs = pair_counts.groupby(pair_columns)[configuration_column].nunique()
        shared_pairs = set(
            shared_pairs[shared_pairs == pair_counts[configuration_column].nunique()].index
        )
        inferential_collisions["shared_eligible_pair"] = [
            (source, target) in shared_pairs
            for source, target in zip(
                inferential_collisions.source_sample_id,
                inferential_collisions.target_sample_id,
            )
        ]
        save_frame(
            inferential_collisions,
            output_path / f"{output_prefix}_collision_attack_records.parquet",
        )
        summaries = []
        group_columns = _configuration_columns(inferential_collisions, "run_dir", "seed", "epsilon")
        for scope, scoped in (
            ("full_eligible", inferential_collisions),
            (
                "shared_intersection",
                inferential_collisions[inferential_collisions.shared_eligible_pair],
            ),
        ):
            for keys, group in scoped.groupby(group_columns, dropna=False):
                key_values = keys if isinstance(keys, tuple) else (keys,)
                summary = {
                    **dict(zip(group_columns, key_values)),
                    "scope": scope,
                    "num_pairs": group[pair_columns].drop_duplicates().shape[0],
                    "collision_success_rate": float(group.collision_success.astype(bool).mean()),
                    "median_distance": float(group.distance.median()),
                    "median_distance_percentile": float(group.distance_percentile.median()),
                }
                for column in (
                    "continuous_collision",
                    "exact_vq_collision",
                    "exact_quantized_collision",
                    "target_class_reached",
                    "vq_token_match_fraction",
                    "quantized_bin_match_fraction",
                ):
                    if column in group:
                        summary[f"mean_{column}"] = float(group[column].dropna().mean())
                summaries.append(summary)
        collision_summary_frame = pd.DataFrame(summaries)
        save_frame(
            collision_summary_frame,
            output_path / f"{output_prefix}_collision_attack_summary.parquet",
        )
    if output_prefix == "phase2" and not frame.empty and "family" in frame:
        # Contrasts are paired within seed and family, with the weakest
        # nominal bottleneck as the preregistered reference level.
        phase2_frame = frame.copy()
        contrast_parts = []
        trend_parts = {}
        for family, family_frame in phase2_frame.groupby("family", dropna=False):
            reference = family_frame[family_frame.bottleneck_strength_ordinal == 0][
                ["seed", "attack", "robust_auc"]
            ].rename(columns={"robust_auc": "robust_auc_weakest"})
            if not reference.empty:
                contrasts = family_frame.merge(reference, on=["seed", "attack"], how="inner")
                contrasts["paired_robust_auc_delta"] = (
                    contrasts.robust_auc - contrasts.robust_auc_weakest
                )
                contrast_parts.append(contrasts)
            for attack, attack_frame in family_frame.groupby("attack"):
                trend_parts[f"{family}:{attack}"] = ordinal_trends(
                    attack_frame, "bottleneck_strength_ordinal", "robust_auc"
                )
        all_contrasts = (
            pd.concat(contrast_parts, ignore_index=True) if contrast_parts else pd.DataFrame()
        )
        save_frame(all_contrasts, output_path / "phase2_paired_contrasts.parquet")
        all_contrasts.to_csv(output_path / "phase2_paired_contrasts.csv", index=False)
        write_json(output_path / "phase2_ordinal_trends.json", trend_parts)
        mean_accuracy = (
            phase2_frame.groupby(["family", "attack", "bottleneck_strength_ordinal"], dropna=False)
            .clean_accuracy.mean()
            .reset_index()
        )
        weakest = mean_accuracy[mean_accuracy.bottleneck_strength_ordinal == 0][
            ["family", "attack", "clean_accuracy"]
        ].rename(columns={"clean_accuracy": "weakest_clean_accuracy"})
        matched = mean_accuracy.merge(weakest, on=["family", "attack"], how="inner")
        matched = matched[(matched.clean_accuracy - matched.weakest_clean_accuracy).abs() <= 0.02]
        save_frame(matched, output_path / "phase2_accuracy_matched_configs.parquet")
        matched_rows = phase2_frame.merge(
            matched[["family", "attack", "bottleneck_strength_ordinal"]],
            on=["family", "attack", "bottleneck_strength_ordinal"],
            how="inner",
        )
        matched_trends = {
            f"{family}:{attack}": ordinal_trends(group, "bottleneck_strength_ordinal", "robust_auc")
            for (family, attack), group in matched_rows.groupby(["family", "attack"])
        }
        write_json(output_path / "phase2_accuracy_matched_trends.json", matched_trends)
        synthesis = phase2_frame[phase2_frame.attack == "input_pgd"].copy()
        if not synthesis.empty:
            synthesis = synthesis.rename(columns={"robust_auc": "local_robustness_auc"})
            latent = phase2_frame[phase2_frame.attack == "latent_pgd"][
                ["run_dir", "robust_auc"]
            ].rename(columns={"robust_auc": "latent_robustness_auc"})
            synthesis = synthesis.merge(latent, on="run_dir", how="left")
            synthesis["semantic_separation_robustness"] = synthesis.get(
                "median_nearest_opposing_distance"
            )
            if not collision_summary_frame.empty:
                full = collision_summary_frame[collision_summary_frame.scope == "full_eligible"]
                collision_auc_rows = []
                for run_name, group in full.groupby("run_dir"):
                    group = group.sort_values("epsilon")
                    x = group.epsilon.to_numpy(dtype=float)
                    y = 1.0 - group.collision_success_rate.to_numpy(dtype=float)
                    auc = (
                        float(np.trapezoid(y, x) / (x.max() - x.min()))
                        if len(x) > 1 and x.max() > x.min()
                        else float(y.mean())
                    )
                    collision_auc_rows.append(
                        {"run_dir": run_name, "collision_resistance_auc": auc}
                    )
                synthesis = synthesis.merge(
                    pd.DataFrame(collision_auc_rows), on="run_dir", how="left"
                )
            save_frame(synthesis, output_path / "phase2_synthesis.parquet")
            synthesis.to_csv(output_path / "phase2_synthesis.csv", index=False)
    return frame


def seed_level_summary(
    frame: pd.DataFrame,
    output_root: str | Path | None = None,
    output_prefix: str = "phase1",
) -> pd.DataFrame:
    """Aggregate configurations while retaining seed-level uncertainty."""
    if frame.empty:
        result = frame.copy()
    else:
        group_columns = [
            column
            for column in ("family", "dz", "beta", "codebook_size", "bits", "attack")
            if column in frame.columns
        ]
        value_columns = [
            column
            for column in ("robust_auc", "clean_accuracy", "median_nearest_opposing_distance")
            if column in frame.columns
        ]
        result = (
            frame.groupby(group_columns, dropna=False)[value_columns]
            .agg(["mean", "std"])
            .reset_index()
        )
        result.columns = [
            (
                "_".join(str(part) for part in column if str(part) != "").rstrip("_")
                if isinstance(column, tuple)
                else str(column)
            )
            for column in result.columns
        ]
    if output_root is not None:
        save_frame(result, Path(output_root) / f"{output_prefix}_seed_summary.parquet")
        result.to_csv(Path(output_root) / f"{output_prefix}_seed_summary.csv", index=False)
    return result


def geometry_correlations(
    summary: pd.DataFrame,
    robustness_column: str = "robust_auc",
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    metrics = [
        "intra_class_distance",
        "inter_class_distance",
        "separation_ratio",
        "median_nearest_opposing_distance",
        "effective_rank",
        "encoder_jacobian_median",
        "contraction_same_median",
        "contraction_different_median",
    ]
    rows = []
    attack_groups = summary.groupby("attack") if "attack" in summary else [(None, summary)]
    rng = np.random.default_rng(seed)
    for attack, attack_frame in attack_groups:
        for metric in metrics:
            required = [metric, robustness_column]
            if "seed" in attack_frame:
                required.append("seed")
            if metric not in attack_frame or robustness_column not in attack_frame:
                continue
            valid = attack_frame[required].dropna()
            if len(valid) < 3:
                continue
            correlation, p_value = spearmanr(valid[metric], valid[robustness_column])
            bootstrap_groups = (
                [group for _, group in valid.groupby("seed")]
                if "seed" in valid
                else [valid.iloc[[index]] for index in range(len(valid))]
            )
            bootstrapped = []
            for _ in range(bootstrap_iterations):
                chosen = rng.integers(0, len(bootstrap_groups), size=len(bootstrap_groups))
                sample = pd.concat([bootstrap_groups[index] for index in chosen])
                if sample[metric].nunique() < 2 or sample[robustness_column].nunique() < 2:
                    continue
                value = spearmanr(sample[metric], sample[robustness_column]).statistic
                if np.isfinite(value):
                    bootstrapped.append(float(value))
            rows.append(
                {
                    "attack": attack,
                    "metric": metric,
                    "rho": correlation,
                    "rho_ci95_lower": (
                        float(np.quantile(bootstrapped, 0.025)) if bootstrapped else float("nan")
                    ),
                    "rho_ci95_upper": (
                        float(np.quantile(bootstrapped, 0.975)) if bootstrapped else float("nan")
                    ),
                    "p_value": p_value,
                    "n": len(valid),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        finite = result.p_value.notna() & np.isfinite(result.p_value)
        result["bh_q_value"] = float("nan")
        p_values = result.loc[finite, "p_value"].to_numpy()
        order = np.argsort(p_values)
        ranked = p_values[order] * len(p_values) / (np.arange(len(p_values)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(len(p_values), dtype=float)
        adjusted[order] = np.minimum(1.0, ranked)
        result.loc[finite, "bh_q_value"] = adjusted
    return result


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    seed_column: str = "seed",
    sample_column: str = "sample_id",
    iterations: int = 2000,
    seed: int = 0,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Bootstrap training seeds first, then observations within each seed."""
    if value_column not in frame:
        raise KeyError(value_column)
    rng = np.random.default_rng(seed)
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    grouped = []
    for _, group in frame.groupby(seed_column):
        if sample_column in group:
            group = group.drop_duplicates(sample_column)
        values = group[value_column].dropna().to_numpy(float)
        if len(values):
            grouped.append(values)
    if not grouped:
        return float("nan"), float("nan"), float("nan")
    estimates = []
    for _ in range(iterations):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        values = []
        for index in chosen:
            group = grouped[index]
            values.extend(group[rng.integers(0, len(group), size=len(group))])
        estimates.append(float(np.mean(values) if statistic == "mean" else np.median(values)))
    return (
        float(np.mean(estimates)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def ordinal_trends(
    frame: pd.DataFrame, strength_column: str, value_column: str
) -> dict[str, float]:
    """Fit linear and quadratic trends against an ordinal bottleneck index."""
    if strength_column not in frame or value_column not in frame:
        return {}
    valid = frame[[strength_column, value_column]].dropna()
    if len(valid) < 3:
        return {}
    ordinal = pd.factorize(valid[strength_column], sort=True)[0].astype(float)
    values = valid[value_column].to_numpy(float)
    linear = np.polyfit(ordinal, values, 1)
    quadratic = np.polyfit(ordinal, values, min(2, len(valid) - 1))
    quadratic = np.pad(quadratic, (3 - len(quadratic), 0))
    return {
        "linear_slope": float(linear[-2]),
        "linear_intercept": float(linear[-1]),
        "linear_rmse": float(np.sqrt(np.mean((np.polyval(linear, ordinal) - values) ** 2))),
        "quadratic_coefficient": float(quadratic[-3]),
        "quadratic_linear_coefficient": float(quadratic[-2]),
        "quadratic_intercept": float(quadratic[-1]),
        "quadratic_rmse": float(np.sqrt(np.mean((np.polyval(quadratic, ordinal) - values) ** 2))),
    }


def validate_phase1_acceptance(output_root: str | Path = "outputs") -> dict[str, Any]:
    """Validate the artifact-level Phase 1 acceptance criteria."""
    output_root = Path(output_root)
    expected = {
        (dimension, seed) for dimension in (512, 256, 128, 64, 32, 16) for seed in (0, 1, 2)
    }
    observed: dict[tuple[int, int], Path] = {}
    identity_runs: list[Path] = []
    errors: list[str] = []
    manifest_hashes: set[str] = set()
    for run_dir in output_root.glob("*/*"):
        if not (run_dir / "COMPLETED").exists():
            continue
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            errors.append(f"missing resolved config: {run_dir}")
            continue
        config = yaml.safe_load(config_path.read_text())
        family = config.get("model", {}).get("family")
        seed = int(config.get("seed", -1))
        if family == "dimensional":
            key = (int(config["model"]["dz"]), seed)
            if key in observed:
                errors.append(f"duplicate dimensional configuration {key}")
            observed[key] = run_dir
        elif family == "identity" and seed == 0:
            identity_runs.append(run_dir)
        hash_path = run_dir / "data_manifest_hash.txt"
        if hash_path.exists():
            manifest_hashes.add(hash_path.read_text().strip())
    for key in sorted(expected - set(observed)):
        errors.append(f"missing dimensional run dz={key[0]} seed={key[1]}")
    if len(identity_runs) != 1:
        errors.append(f"expected one seed-0 identity diagnostic, found {len(identity_runs)}")
    if len(manifest_hashes) != 1:
        errors.append(f"expected one shared manifest hash, found {len(manifest_hashes)}")
    for run_dir in [*observed.values(), *identity_runs]:
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
        input_zero_predictions: pd.DataFrame | None = None
        manifest_path = Path(config["data"]["manifest"])
        if not manifest_path.exists():
            errors.append(f"missing data manifest: {manifest_path}")
            final_ids: set[str] = set()
        else:
            manifest = pd.read_csv(manifest_path)
            final_ids = set(manifest.loc[manifest.split == "final", "sample_id"].astype(str))
        history_path = run_dir / "history.parquet"
        if not history_path.exists():
            errors.append(f"missing history: {run_dir}")
        elif pd.read_parquet(history_path).select_dtypes(include=[np.number]).isna().any().any():
            errors.append(f"NaN in history: {run_dir}")
        for required in (run_dir / "artifacts" / "latents_final.safetensors",):
            if not required.exists():
                errors.append(f"missing artifact: {required}")
        for kind, filename, legacy_filename in (
            ("geometry", "geometry.json", "geometry_final.json"),
            ("invariance", "invariance.json", "invariance_final.json"),
        ):
            try:
                analysis_path = _completed_analysis_file(
                    run_dir, kind, "final", filename, legacy_filename
                )
                if analysis_path is None:
                    raise FileNotFoundError(filename)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"missing or ambiguous {kind} analysis: {run_dir}: {exc}")
        completed_evaluations = []
        for candidate in (run_dir / "evaluations").glob("robustness-*-attempt*"):
            config_path = candidate / "config.json"
            if (
                (candidate / "COMPLETED").exists()
                and (candidate / "input_pgd.parquet").exists()
                and (candidate / "latent_pgd.parquet").exists()
                and config_path.exists()
                and json.loads(config_path.read_text()).get("attack_protocol_version")
                == ATTACK_PROTOCOL_VERSION
            ):
                completed_evaluations.append(candidate)
        if len(completed_evaluations) != 1:
            errors.append(
                f"expected exactly one completed robustness evaluation for {run_dir}, "
                f"found {len(completed_evaluations)}"
            )
        elif final_ids:
            evaluation_dir = completed_evaluations[0]
            expected_radii = {
                "input_pgd.parquet": {float(value) for value in config["attack"]["input_epsilons"]},
                "latent_pgd.parquet": {
                    0.0,
                    *(float(value) for value in config["attack"]["latent_rhos"]),
                },
            }
            for filename, radii in expected_radii.items():
                records = pd.read_parquet(evaluation_dir / filename)
                if set(records.sample_id.astype(str)) != final_ids:
                    errors.append(f"incomplete final sample IDs in {evaluation_dir / filename}")
                observed_radii = set(records.radius.astype(float))
                if observed_radii != radii:
                    errors.append(f"incorrect radius grid in {evaluation_dir / filename}")
                counts = records.groupby("radius").sample_id.nunique()
                if not counts.eq(len(final_ids)).all():
                    errors.append(f"incomplete per-radius records in {evaluation_dir / filename}")
                if "attack_protocol_version" not in records or set(
                    records.attack_protocol_version.astype(int)
                ) != {ATTACK_PROTOCOL_VERSION}:
                    errors.append(f"stale attack protocol in {evaluation_dir / filename}")
                if (
                    filename == "input_pgd.parquet"
                    and (records.linf_norm > records.radius.astype(float) + 1e-6).any()
                ):
                    errors.append(f"input attack bound violation in {evaluation_dir / filename}")
                if (
                    filename == "latent_pgd.parquet"
                    and (records.relative_l2_norm > records.radius.astype(float) + 1e-5).any()
                ):
                    errors.append(f"latent attack bound violation in {evaluation_dir / filename}")
                if filename == "input_pgd.parquet":
                    input_zero_predictions = records.loc[
                        records.radius.astype(float).eq(0.0),
                        ["sample_id", "clean_prediction"],
                    ].drop_duplicates("sample_id")
        clean_evaluations = []
        for candidate in (run_dir / "evaluations").glob("clean-*-attempt*"):
            record_path = candidate / "clean_final.parquet"
            summary_path = candidate / "clean_final.json"
            if not (candidate / "COMPLETED").exists() or not record_path.exists():
                continue
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if summary.get("max_samples") is None and summary.get("split") == "final":
                clean_evaluations.append(record_path)
        if len(clean_evaluations) != 1:
            errors.append(
                f"expected exactly one completed clean evaluation for {run_dir}, "
                f"found {len(clean_evaluations)}"
            )
        elif final_ids:
            clean = pd.read_parquet(clean_evaluations[0])
            if set(clean.sample_id.astype(str)) != final_ids:
                errors.append(f"incomplete clean sample IDs in {clean_evaluations[0]}")
            if input_zero_predictions is not None:
                comparison = clean[["sample_id", "prediction"]].merge(
                    input_zero_predictions, on="sample_id", how="inner", validate="one_to_one"
                )
                mismatch_count = int(comparison.prediction.ne(comparison.clean_prediction).sum())
                if mismatch_count:
                    errors.append(
                        "input radius-zero predictions disagree with clean evaluation for "
                        f"{mismatch_count} samples: {run_dir}"
                    )
        diagnostics = [
            candidate
            for candidate in (run_dir / "evaluations").glob("attack-diagnostics-*-attempt*")
            if (candidate / "COMPLETED").exists()
            and (candidate / "diagnostics.json").exists()
            and json.loads((candidate / "diagnostics.json").read_text()).get(
                "attack_protocol_version"
            )
            == ATTACK_PROTOCOL_VERSION
        ]
        if len(diagnostics) != 1:
            errors.append(
                f"expected exactly one attack diagnostic for {run_dir}, found {len(diagnostics)}"
            )
        elif json.loads((diagnostics[0] / "diagnostics.json").read_text())["status"] != "passed":
            errors.append(f"failed attack diagnostics: {run_dir}")
    identity_comparison = output_root / "phase1_identity_diagnostic.parquet"
    if not identity_comparison.exists():
        errors.append("missing identity-versus-dimensional-512 diagnostic comparison")
    report = {
        "status": "ready" if not errors else "not_ready",
        "expected_dimensional_runs": len(expected),
        "observed_dimensional_runs": len(observed),
        "identity_runs": len(identity_runs),
        "errors": errors,
    }
    write_json(output_root / "phase1_acceptance.json", report)
    return report


def _valid_superseding_attack_audit(
    run_dir: Path,
    diagnostics_path: Path,
    amendment_path: Path,
    amendment: dict[str, Any],
) -> tuple[bool, str]:
    """Validate one amended audit's immutable provenance and sample manifest."""
    try:
        report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        config_path = run_dir / "resolved_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        protocol = amendment["protocol"]
        required = {
            "audit_amendment_id": amendment["amendment_id"],
            "audit_amendment_sha256": sha256_file(amendment_path),
            "attack_protocol_version": int(protocol["attack_protocol_version"]),
            "diagnostic_samples": int(protocol["sample_count"]),
            "tolerance": float(protocol["tolerance"]),
            "baseline_steps": int(protocol["baseline"]["steps"]),
            "baseline_restarts": int(protocol["baseline"]["restarts"]),
            "stronger_steps": int(protocol["stronger"]["steps"]),
            "stronger_restarts": int(protocol["stronger"]["restarts"]),
            "sample_selection": protocol["sample_selection"],
            "status": "passed",
            "independent_audit": True,
        }
        for key, expected in required.items():
            observed = report.get(key)
            if isinstance(expected, float):
                if observed is None or abs(float(observed) - expected) > 1e-12:
                    return False, f"{key} mismatch"
            elif observed != expected:
                return False, f"{key} mismatch"
        checkpoint = str(report.get("checkpoint", ""))
        checkpoint_path = run_dir / "checkpoints" / checkpoint
        if not checkpoint_path.is_file():
            return False, "checkpoint is missing"
        provenance = {
            "resolved_config_sha256": sha256_file(config_path),
            "data_manifest_sha256": sha256_file(Path(config["data"]["manifest"])),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        for key, expected in provenance.items():
            if report.get(key) != expected:
                return False, f"{key} mismatch"
        sample_ids = [str(value) for value in report.get("sample_ids", [])]
        if len(sample_ids) != int(protocol["sample_count"]):
            return False, "sample count mismatch"
        if report.get("sample_manifest_hash") != sample_ids_hash(sample_ids):
            return False, "sample manifest hash mismatch"
        loader = make_loader(config, "final", batch_size=1)
        indices = _stratified_dataset_indices(loader.dataset, len(sample_ids))
        expected_ids = [str(loader.dataset.frame.iloc[index].sample_id) for index in indices]
        if sample_ids != expected_ids:
            return False, "sample manifest contents mismatch"
        if str(config.get("model", {}).get("family", "")).lower() == "vib":
            if report.get("baseline_eot_samples") != int(protocol["vib"]["baseline_eot_samples"]):
                return False, "VIB baseline EoT mismatch"
            if report.get("increased_eot_samples") != int(protocol["vib"]["stronger_eot_samples"]):
                return False, "VIB stronger EoT mismatch"
        return True, "valid"
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        return False, str(exc)


def validate_phase2_acceptance(
    output_root: str | Path = "outputs",
    runtime_amendment: str | Path | None = None,
    audit_amendment: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the final artifact contract for the four new Phase 2 families."""
    output_root = Path(output_root)
    amendment = None
    if runtime_amendment is not None:
        from .phase2 import load_runtime_amendment

        amendment = load_runtime_amendment(runtime_amendment)
    audit_protocol = None
    audit_amendment_path = None
    if audit_amendment is not None:
        from .phase2 import load_attack_audit_amendment

        audit_amendment_path = Path(audit_amendment)
        audit_protocol = load_attack_audit_amendment(audit_amendment_path)
    expected = {
        (family, value, seed)
        for family in ("vib", "vq", "quantized", "autoencoder")
        for value in PHASE2_SPECS[family][1]
        for seed in (0, 1, 2)
    }
    observed: dict[tuple[str, Any, int], Path] = {}
    errors: list[str] = []
    decision_records = sorted(Path("configs").glob("phase2_decision_*.yaml"))
    if not decision_records:
        errors.append("missing timestamped frozen Phase 2 decision record")
    else:
        for decision_record in decision_records:
            try:
                validate_decision_record(decision_record)
            except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                errors.append(f"invalid Phase 2 decision record {decision_record}: {exc}")
    canonical_runs, duplicate_groups = canonical_completed_runs(
        output_root, ("vib", "vq", "quantized", "autoencoder")
    )
    canonical_set = {str(path) for path in canonical_runs}
    if duplicate_groups and not list(output_root.glob("phase2_duplicate_provenance_*.json")):
        errors.append("duplicate Phase 2 attempts lack a provenance audit")
    for run_dir in output_root.glob("*/*"):
        if not (run_dir / "COMPLETED").exists():
            continue
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        family = str(config.get("model", {}).get("family", "")).lower()
        if family not in {key[0] for key in expected}:
            continue
        if (
            family in {"vib", "vq", "quantized", "autoencoder"}
            and str(run_dir) not in canonical_set
        ):
            continue
        parameter = PHASE2_SPECS[family][0]
        value = config.get("model", {}).get(parameter)
        if family == "quantized":
            value = (
                "FP32" if value is None or str(value).lower() in {"fp32", "none"} else int(value)
            )
        elif family == "vib":
            value = float(value)
        else:
            value = int(value)
        key = (family, value, int(config.get("seed", -1)))
        observed[key] = run_dir
    for key in sorted(expected, key=lambda item: (item[0], str(item[1]), item[2])):
        if key not in observed:
            errors.append(f"missing Phase 2 run family={key[0]} strength={key[1]} seed={key[2]}")
    for key, run_dir in observed.items():
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if amendment is None:
            policy = {
                "phase2_robustness": True,
                "phase2_masking_checks": key[0] in {"vq", "quantized"},
            }
        else:
            from .phase2 import runtime_policy_for_config

            policy = runtime_policy_for_config(config, amendment)
        decision_reference = config.get("phase2_decision_record")
        if not isinstance(decision_reference, dict):
            errors.append(f"missing embedded Phase 2 decision record: {run_dir}")
        else:
            decision_path = Path(decision_reference.get("path", ""))
            if not decision_path.exists():
                errors.append(f"missing embedded decision record path: {run_dir}")
            else:
                try:
                    decision = validate_decision_record(decision_path)
                    if decision_reference.get("sha256") != decision_record_hash(decision_path):
                        errors.append(f"decision record hash mismatch: {run_dir}")
                    if decision_reference.get("contents") != decision:
                        errors.append(f"embedded decision record contents mismatch: {run_dir}")
                except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                    errors.append(f"invalid embedded decision record: {run_dir}: {exc}")
        if not (run_dir / "environment.json").exists():
            errors.append(f"missing environment metadata: {run_dir}")
        manifest_hash_path = run_dir / "data_manifest_hash.txt"
        if not manifest_hash_path.exists():
            errors.append(f"missing manifest hash: {run_dir}")
        elif Path(
            config["data"]["manifest"]
        ).exists() and manifest_hash_path.read_text().strip() != manifest_hash(
            config["data"]["manifest"]
        ):
            errors.append(f"manifest hash mismatch: {run_dir}")
        for required in (
            run_dir / "history.parquet",
            run_dir / "artifacts" / "latents_development_tune.safetensors",
            run_dir / "artifacts" / "latents_development_tune_index.parquet",
            run_dir / "artifacts" / "latents_final.safetensors",
            run_dir / "artifacts" / "latents_final_index.parquet",
        ):
            if not required.exists():
                errors.append(f"missing Phase 2 artifact: {required}")
        for kind, filename, legacy_filename in (
            ("geometry", "geometry.json", "geometry_final.json"),
            ("invariance", "invariance.json", "invariance_final.json"),
        ):
            try:
                analysis_path = _completed_analysis_file(
                    run_dir, kind, "final", filename, legacy_filename
                )
                if analysis_path is None:
                    raise FileNotFoundError(filename)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"missing or ambiguous Phase 2 {kind} analysis: {run_dir}: {exc}")
        if key[0] == "autoencoder":
            clean_prefix = "autoencoder-clean-"
            robustness_prefix = "autoencoder-robustness-"
        else:
            clean_prefix = "clean-"
            robustness_prefix = "robustness-"
        files = (
            ("input_pgd_reference.parquet", "latent_pgd_reference.parquet")
            if key[0] == "autoencoder"
            else ("input_pgd.parquet", "latent_pgd.parquet")
        )
        records_by_name = {}
        completed = [
            candidate
            for candidate in (run_dir / "evaluations").glob(f"{robustness_prefix}*-attempt*")
            if (candidate / "COMPLETED").exists() and (candidate / "config.json").exists()
        ]
        final_robustness = [
            candidate
            for candidate in completed
            if (json.loads((candidate / "config.json").read_text()).get("split") == "final")
            and json.loads((candidate / "config.json").read_text()).get("attack_protocol_version")
            == ATTACK_PROTOCOL_VERSION
        ]
        if (
            not final_robustness and policy["phase2_robustness"]
        ) or len(final_robustness) > 1:
            errors.append(f"expected one final robustness evaluation: {run_dir}")
        elif final_robustness:
            evaluation = final_robustness[0]
            evaluation_config = json.loads((evaluation / "config.json").read_text())
            if evaluation_config.get("max_samples") is not None:
                errors.append(f"final robustness evaluation is truncated: {evaluation}")
            for filename in files:
                if not (evaluation / filename).exists():
                    errors.append(f"missing final attack records: {evaluation / filename}")
                else:
                    records_by_name[filename] = pd.read_parquet(evaluation / filename)
            if len(records_by_name) == 2:
                manifest_path = Path(config["data"]["manifest"])
                final_ids = set()
                if manifest_path.exists():
                    manifest = pd.read_csv(manifest_path)
                    final_ids = set(
                        manifest.loc[manifest.split == "final", "sample_id"].astype(str)
                    )
                expected_radii = {
                    files[0]: {float(value) for value in config["attack"]["input_epsilons"]},
                    files[1]: {0.0, *(float(value) for value in config["attack"]["latent_rhos"])},
                }
                for filename, records in records_by_name.items():
                    if final_ids and set(records.sample_id.astype(str)) != final_ids:
                        errors.append(f"incomplete final sample IDs: {evaluation / filename}")
                    if set(records.radius.astype(float)) != expected_radii[filename]:
                        errors.append(f"incorrect final radius grid: {evaluation / filename}")
                    counts = records.groupby("radius").sample_id.nunique()
                    if final_ids and not counts.eq(len(final_ids)).all():
                        errors.append(
                            f"incomplete final per-radius records: {evaluation / filename}"
                        )
                    if "attack_protocol_version" not in records or set(
                        records.attack_protocol_version.astype(int)
                    ) != {ATTACK_PROTOCOL_VERSION}:
                        errors.append(f"stale attack protocol: {evaluation / filename}")
                    if (
                        filename == files[0]
                        and (records.linf_norm > records.radius.astype(float) + 1e-6).any()
                    ):
                        errors.append(f"input attack bound violation: {evaluation / filename}")
                    if (
                        filename == files[1]
                        and (records.relative_l2_norm > records.radius.astype(float) + 1e-5).any()
                    ):
                        errors.append(f"latent attack bound violation: {evaluation / filename}")
            if policy["phase2_masking_checks"]:
                square = [
                    candidate
                    for candidate in (run_dir / "evaluations").glob("square-*-attempt*")
                    if (candidate / "COMPLETED").exists()
                    and (candidate / "square_attack.parquet").exists()
                    and (candidate / "config.json").exists()
                ]
                square = [
                    candidate
                    for candidate in square
                    if json.loads((candidate / "config.json").read_text()).get("split") == "final"
                    and json.loads((candidate / "config.json").read_text()).get(
                        "attack_protocol_version"
                    )
                    == ATTACK_PROTOCOL_VERSION
                ]
                if len(square) != 1:
                    errors.append(f"expected one final Square Attack evaluation: {run_dir}")
                elif (
                    json.loads((square[0] / "config.json").read_text()).get("max_samples")
                    != 1000
                ):
                    errors.append(f"Square Attack does not use the registered subset: {run_dir}")
                transfer = [
                    candidate
                    for candidate in (run_dir / "evaluations").glob("transfer-*-attempt*")
                    if (candidate / "COMPLETED").exists()
                    and (candidate / "transfer_attack.parquet").exists()
                    and (candidate / "config.json").exists()
                    and json.loads((candidate / "config.json").read_text()).get("split") == "final"
                    and json.loads((candidate / "config.json").read_text()).get(
                        "attack_protocol_version"
                    )
                    == ATTACK_PROTOCOL_VERSION
                ]
                if len(transfer) != 1:
                    errors.append(f"expected one final nearest-capacity transfer attack: {run_dir}")
                elif (
                    json.loads((transfer[0] / "config.json").read_text()).get("max_samples") != 1000
                ):
                    errors.append(f"transfer attack does not use the registered subset: {run_dir}")
        if amendment is None:
            collision = [
                candidate
                for candidate in (run_dir / "evaluations").glob("collision-*-attempt*")
                if (candidate / "COMPLETED").exists()
                and (candidate / "collision_attacks.parquet").exists()
                and (candidate / "config.json").exists()
                and json.loads((candidate / "config.json").read_text()).get("split") == "final"
                and json.loads((candidate / "config.json").read_text()).get(
                    "attack_protocol_version"
                )
                == ATTACK_PROTOCOL_VERSION
            ]
            if len(collision) != 1:
                errors.append(f"expected one final collision evaluation: {run_dir}")
            else:
                collision_config = json.loads((collision[0] / "config.json").read_text())
                tuning_artifact_value = collision_config.get("tuning_artifact")
                tuning_artifact = Path(tuning_artifact_value) if tuning_artifact_value else None
                if (
                    tuning_artifact is None
                    or not tuning_artifact.is_file()
                    or collision_config.get("tuning_artifact_sha256")
                    != sha256_file(tuning_artifact)
                ):
                    errors.append(f"collision tuning artifact is missing or changed: {run_dir}")
                collision_records_frame = pd.read_parquet(
                    collision[0] / "collision_attacks.parquet"
                )
                expected_collision_radii = {
                    0.0,
                    *(float(value) for value in config["attack"]["input_epsilons"]),
                }
                if (
                    set(collision_records_frame.epsilon.astype(float))
                    != expected_collision_radii
                ):
                    errors.append(f"incorrect collision epsilon grid: {run_dir}")
                if (
                    "input_bound_satisfied" not in collision_records_frame
                    or not collision_records_frame.input_bound_satisfied.astype(bool).all()
                ):
                    errors.append(f"collision input-bound check failed: {run_dir}")
        clean = []
        for candidate in (run_dir / "evaluations").glob(f"{clean_prefix}*-attempt*"):
            if not (candidate / "COMPLETED").exists():
                continue
            summary_path = (
                candidate / "reconstruction_final.json"
                if key[0] == "autoencoder"
                else candidate / "clean_final.json"
            )
            record_path = (
                candidate / "reconstruction_final.parquet"
                if key[0] == "autoencoder"
                else candidate / "clean_final.parquet"
            )
            if not summary_path.exists() or not record_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if key[0] == "autoencoder" and not {
                "reference_original_accuracy",
                "reconstruction_accuracy",
                "reconstruction_mse",
                "reconstruction_psnr_db",
            }.issubset(summary):
                errors.append(f"incomplete autoencoder reconstruction summary: {summary_path}")
                continue
            if summary.get("split") == "final" and summary.get("max_samples") is None:
                clean.append((record_path, summary))
        if len(clean) != 1:
            if not clean:
                errors.append(f"missing final clean evaluation: {run_dir}")
            else:
                errors.append(f"expected one final clean evaluation: {run_dir}")
        if len(clean) == 1 and len(records_by_name) == 2:
            clean_frame = pd.read_parquet(clean[0][0])
            input_frame = records_by_name[files[0]]
            zero = input_frame[input_frame.radius.astype(float).eq(0.0)]
            if (
                not zero.empty
                and "adversarial_prediction" in zero
                and zero.clean_prediction.ne(zero.adversarial_prediction).any()
            ):
                errors.append(f"radius-zero attack changes predictions: {run_dir}")
            prediction_column = (
                "reconstruction_prediction" if key[0] == "autoencoder" else "prediction"
            )
            # VIB clean and attack evaluations independently average finite
            # posterior samples. Near-tied examples can differ even when both
            # use the registered 32 draws, so validate the attack's internal
            # epsilon-zero baseline rather than demanding cross-run identity.
            if key[0] != "vib" and prediction_column in clean_frame and not zero.empty:
                comparison = clean_frame[["sample_id", prediction_column]].merge(
                    zero[["sample_id", "clean_prediction"]],
                    on="sample_id",
                    how="inner",
                    validate="one_to_one",
                )
                if comparison[prediction_column].ne(comparison.clean_prediction).any():
                    errors.append(f"radius-zero predictions disagree with clean: {run_dir}")
        diagnostics = [
            candidate
            for candidate in (run_dir / "evaluations").glob("attack-diagnostics-*-attempt*")
            if (candidate / "COMPLETED").exists()
            and (candidate / "diagnostics.json").exists()
            and json.loads((candidate / "diagnostics.json").read_text()).get(
                "attack_protocol_version"
            )
            == ATTACK_PROTOCOL_VERSION
        ]
        if key[0] == "autoencoder":
            diagnostics += [
                candidate
                for candidate in (run_dir / "evaluations").glob(
                    "autoencoder-attack-diagnostics-*-attempt*"
                )
                if (candidate / "COMPLETED").exists()
                and (candidate / "diagnostics.json").exists()
                and json.loads((candidate / "diagnostics.json").read_text()).get(
                    "attack_protocol_version"
                )
                == ATTACK_PROTOCOL_VERSION
            ]
        passed_diagnostics = [
            candidate
            for candidate in diagnostics
            if json.loads((candidate / "diagnostics.json").read_text()).get("status") == "passed"
            and not json.loads((candidate / "diagnostics.json").read_text()).get(
                "independent_audit", False
            )
        ]
        if audit_protocol is not None and audit_amendment_path is not None:
            for candidate in diagnostics:
                diagnostic_report = json.loads((candidate / "diagnostics.json").read_text())
                if diagnostic_report.get("audit_amendment_id") != audit_protocol["amendment_id"]:
                    continue
                valid, reason = _valid_superseding_attack_audit(
                    run_dir, candidate / "diagnostics.json", audit_amendment_path, audit_protocol
                )
                if valid:
                    passed_diagnostics.append(candidate)
                else:
                    errors.append(f"invalid superseding attack audit: {candidate}: {reason}")
        if policy["phase2_robustness"] and not passed_diagnostics:
            if diagnostics:
                errors.append(f"failed attack correctness diagnostics: {run_dir}")
            else:
                errors.append(f"expected one attack correctness diagnostic: {run_dir}")
    report = {
        "status": "ready" if not errors else "not_ready",
        "expected_runs": len(expected),
        "observed_runs": len(observed),
        "decision_records": [str(path) for path in decision_records],
        "runtime_amendment": str(runtime_amendment) if runtime_amendment is not None else None,
        "audit_amendment": str(audit_amendment) if audit_amendment is not None else None,
        "errors": errors,
    }
    write_json(output_root / "phase2_acceptance.json", report)
    return report


def analyze_identity_followup(
    output_root: str | Path = "outputs", report_path: str | Path | None = None
) -> dict[str, Any]:
    """Audit the required three-seed identity control without overwriting runs.

    When all three controls and their final records exist, a timestamped
    identity-vs-learned-512 aggregate is produced. With missing seeds the
    immutable status report records the gap and leaves the prior diagnostic
    untouched.
    """
    output_root = Path(output_root)
    identity_runs = []
    for run_dir in (output_root / "identity").glob("*"):
        if not (run_dir / "COMPLETED").exists():
            continue
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("model", {}).get("family") == "identity":
            identity_runs.append((int(config.get("seed", -1)), run_dir))
    seeds = sorted(seed for seed, _ in identity_runs)
    duplicate_seeds = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
    missing = sorted({0, 1, 2} - set(seeds))
    artifact_gaps = []
    for seed, run_dir in identity_runs:
        required = (
            run_dir / "artifacts" / "latents_final.safetensors",
            run_dir / "artifacts" / "latents_final_index.parquet",
        )
        analyses_present = True
        for kind, filename, legacy_filename in (
            ("geometry", "geometry.json", "geometry_final.json"),
            ("invariance", "invariance.json", "invariance_final.json"),
        ):
            try:
                analysis_path = _completed_analysis_file(
                    run_dir, kind, "final", filename, legacy_filename
                )
                if analysis_path is None:
                    raise FileNotFoundError(filename)
            except (FileNotFoundError, ValueError):
                analyses_present = False
        if any(not path.exists() for path in required) or not analyses_present:
            artifact_gaps.append({"seed": seed, "run_dir": str(run_dir)})
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if report_path is None:
        report_path = output_root / f"phase1_identity_followup_{timestamp}.json"
    report = {
        "status": (
            "ready" if not missing and not duplicate_seeds and not artifact_gaps else "blocked"
        ),
        "expected_seeds": [0, 1, 2],
        "completed_seeds": seeds,
        "missing_seeds": missing,
        "duplicate_seeds": duplicate_seeds,
        "artifact_gaps": artifact_gaps,
        "prior_diagnostic_preserved": (output_root / "phase1_identity_diagnostic.parquet").exists(),
    }
    if report["status"] == "ready":
        aggregate_prefix = f"phase1_identity_followup_{timestamp}"
        aggregate_completed_runs(
            output_root,
            output_prefix=aggregate_prefix,
            include_families=["identity", "dimensional"],
        )
        report["aggregate_prefix"] = aggregate_prefix
    write_json(Path(report_path), report)
    return report
