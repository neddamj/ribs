"""Analysis, aggregation, and statistical reporting from saved artifacts."""

from __future__ import annotations

import hashlib
import json
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
from .data import manifest_hash
from .evaluation import _ReconstructionTask, extract_latents, load_model, make_loader
from .geometry import contraction_metrics, encoder_spectral_norm, geometry_metrics
from .invariance import invariance_metrics
from .metrics import shared_clean_correct, summarize_curve
from .utils import write_json


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
    latent_path = run_dir / "artifacts" / f"latents_{split}.safetensors"
    index_path = run_dir / "artifacts" / f"latents_{split}_index.parquet"
    tensors = load_latents(latent_path)
    index = pd.read_parquet(index_path)
    if max_samples is not None:
        tensors = {key: value[:max_samples] for key, value in tensors.items()}
        index = index.iloc[:max_samples].reset_index(drop=True)
    labels = torch.as_tensor(index.label.to_numpy())
    metrics = geometry_metrics(tensors["canonical_latent"], labels)
    model, config, device = load_model(run_dir)
    loader = make_loader(config, split)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(config["data"].get("image_size", 224))
    raw_path = analysis_dir / f".raw_images_{split}.dat"
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
    write_json(run_dir / "analysis" / f"geometry_{split}.json", metrics)
    distances, indices = nearest_opposing(tensors["canonical_latent"], labels)
    collision = natural_collision_metrics(tensors["canonical_latent"], labels)
    tuning_path = run_dir / "artifacts" / "latents_development_tune.safetensors"
    tuning_index_path = run_dir / "artifacts" / "latents_development_tune_index.parquet"
    if not tuning_path.exists() or not tuning_index_path.exists():
        extract_latents(run_dir, "development_tune")
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
    save_frame(collision_frame, run_dir / "analysis" / f"natural_collisions_{split}.parquet")
    write_json(run_dir / "analysis" / f"natural_collision_summary_{split}.json", collision)
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
    model, config, device = load_model(run_dir)
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
        model = _ReconstructionTask(model, reference).to(device).eval()
    loader = make_loader(config, split)
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
    latent_path = Path(run_dir) / "artifacts" / f"latents_{split}.safetensors"
    index_path = Path(run_dir) / "artifacts" / f"latents_{split}_index.parquet"
    if not latent_path.exists() or not index_path.exists():
        extract_latents(run_dir, split)
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
    write_json(Path(run_dir) / "analysis" / f"invariance_{split}.json", result)
    return result


def aggregate_completed_runs(output_root: str | Path = "outputs") -> pd.DataFrame:
    rows = []
    curve_rows = []
    sample_rows = []
    distance_rows = []
    collision_attack_rows = []
    clean_records: dict[str, pd.DataFrame] = {}
    completed_runs = [
        path for path in Path(output_root).glob("*/*") if (path / "COMPLETED").exists()
    ]
    training_configs: dict[str, list[Path]] = {}
    for run_dir in completed_runs:
        config_path = run_dir / "resolved_config.yaml"
        if config_path.exists():
            comparable = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            comparable.pop("resume", None)
            if "phase2_decision_record" in comparable:
                comparable["phase2_decision_record"].pop("path", None)
            digest = hashlib.sha256(
                json.dumps(comparable, sort_keys=True, default=str).encode()
            ).hexdigest()
            training_configs.setdefault(digest, []).append(run_dir)
    duplicates = {key: value for key, value in training_configs.items() if len(value) > 1}
    if duplicates:
        names = [[str(path) for path in paths] for paths in duplicates.values()]
        raise ValueError(f"Duplicate completed attempts require explicit resolution: {names}")
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
        evaluation_dirs = []
        candidates = [
            *(run_dir / "evaluations").glob("robustness-*-attempt*"),
            *(run_dir / "evaluations").glob("autoencoder-robustness-*-attempt*"),
        ]
        for candidate in candidates:
            config_file = candidate / "config.json"
            if not (candidate / "COMPLETED").exists() or not config_file.exists():
                continue
            evaluation_config = json.loads(config_file.read_text())
            if (
                evaluation_config.get("split") == "final"
                and evaluation_config.get("max_samples") is None
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
            ]
            if len(square_dirs) > 1:
                raise ValueError(f"Multiple completed Square Attack evaluations for {run_dir}")
            if square_dirs:
                square = pd.read_parquet(square_dirs[0] / "square_attack.parquet")
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
        geometry_path = run_dir / "analysis" / "geometry_final.json"
        if geometry_path.exists():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            for row in rows:
                if row["run_dir"] == str(run_dir):
                    row.update(geometry)
        collision_path = run_dir / "analysis" / "natural_collisions_final.parquet"
        if collision_path.exists():
            natural_records = pd.read_parquet(collision_path)
            distance_rows.extend(
                {**base, **record}
                for record in natural_records[["sample_id", "distance"]].to_dict("records")
            )
        collision_dirs = [
            candidate
            for candidate in (run_dir / "evaluations").glob("collision-*-attempt*")
            if (candidate / "COMPLETED").exists()
            and (candidate / "collision_attacks.parquet").exists()
        ]
        if len(collision_dirs) > 1:
            raise ValueError(f"Multiple completed collision evaluations found for {run_dir}")
        if collision_dirs:
            collision_attacks = pd.read_parquet(collision_dirs[0] / "collision_attacks.parquet")
            collision_attack_rows.extend(
                {**base, **record} for record in collision_attacks.to_dict("records")
            )
    frame = pd.DataFrame(rows)
    save_frame(frame, Path(output_root) / "phase1_summary.parquet")
    frame.to_csv(Path(output_root) / "phase1_summary.csv", index=False)
    save_frame(pd.DataFrame(curve_rows), Path(output_root) / "phase1_curves.parquet")
    if clean_records:
        save_frame(
            shared_clean_correct(clean_records),
            Path(output_root) / "phase1_shared_clean_correct.parquet",
        )
    inferential_frame = frame[
        ~frame.get("codebook_collapsed", pd.Series(False, index=frame.index)).fillna(False)
    ].copy()
    excluded = frame.loc[
        frame.index.difference(inferential_frame.index),
        [column for column in ("run_dir", "family", "seed") if column in frame],
    ]
    write_json(
        Path(output_root) / "phase1_inferential_exclusions.json",
        {
            "reason": "codebook collapse (<10% active codes for five consecutive epochs)",
            "runs": excluded.to_dict("records"),
        },
    )
    seed_level_summary(inferential_frame, output_root)
    primary_geometry = (
        inferential_frame[inferential_frame.family == "dimensional"]
        if "family" in inferential_frame
        else inferential_frame
    )
    correlations = geometry_correlations(primary_geometry)
    save_frame(correlations, Path(output_root) / "phase1_geometry_correlations.parquet")
    correlations.to_csv(Path(output_root) / "phase1_geometry_correlations.csv", index=False)
    dimensional = inferential_frame[
        inferential_frame.get("family", pd.Series(index=inferential_frame.index)) == "dimensional"
    ].copy()
    if not dimensional.empty and {"dz", "seed", "attack", "robust_auc"}.issubset(dimensional):
        baseline = dimensional[dimensional.dz == 512][["seed", "attack", "robust_auc"]].rename(
            columns={"robust_auc": "robust_auc_dz512"}
        )
        contrasts = dimensional.merge(baseline, on=["seed", "attack"], how="inner")
        contrasts["paired_robust_auc_delta"] = contrasts.robust_auc - contrasts.robust_auc_dz512
        save_frame(contrasts, Path(output_root) / "phase1_paired_contrasts.parquet")
        contrasts.to_csv(Path(output_root) / "phase1_paired_contrasts.csv", index=False)
        mean_accuracy = dimensional.groupby(["attack", "dz"]).clean_accuracy.mean().reset_index()
        reference_accuracy = mean_accuracy[mean_accuracy.dz == 512][
            ["attack", "clean_accuracy"]
        ].rename(columns={"clean_accuracy": "reference_clean_accuracy"})
        matched = mean_accuracy.merge(reference_accuracy, on="attack")
        matched = matched[(matched.clean_accuracy - matched.reference_clean_accuracy).abs() <= 0.02]
        save_frame(matched, Path(output_root) / "phase1_accuracy_matched_configs.parquet")
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
        write_json(Path(output_root) / "phase1_ordinal_trends.json", trends)
        matched_dimensions = matched[["attack", "dz"]].drop_duplicates()
        matched_rows = dimensional.merge(matched_dimensions, on=["attack", "dz"], how="inner")
        matched_trends = {
            attack: ordinal_trends(group, "bottleneck_strength", "robust_auc")
            for attack, group in matched_rows.groupby("attack")
        }
        write_json(
            Path(output_root) / "phase1_accuracy_matched_trends.json",
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
            Path(output_root) / "phase1_identity_diagnostic.parquet",
        )
    sample_frame = pd.DataFrame(sample_rows)
    bootstrap_rows = []
    if not sample_frame.empty:
        sample_frame = sample_frame[~sample_frame["codebook_collapsed"].fillna(False).astype(bool)]
        group_columns = [
            column for column in ("family", "dz", "attack", "radius") if column in sample_frame
        ]
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
        group_columns = [column for column in ("family", "dz") if column in distance_frame]
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
        pd.DataFrame(bootstrap_rows), Path(output_root) / "phase1_hierarchical_bootstrap.parquet"
    )
    collision_frame = pd.DataFrame(collision_attack_rows)
    if not collision_frame.empty:
        save_frame(
            collision_frame,
            Path(output_root) / "collision_attack_records_all.parquet",
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
            Path(output_root) / "collision_attack_records.parquet",
        )
        summaries = []
        group_columns = [
            column
            for column in ("family", "dz", "beta", "codebook_size", "bits", "epsilon")
            if column in inferential_collisions
        ]
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
        save_frame(pd.DataFrame(summaries), Path(output_root) / "collision_attack_summary.parquet")
    return frame


def seed_level_summary(frame: pd.DataFrame, output_root: str | Path | None = None) -> pd.DataFrame:
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
        save_frame(result, Path(output_root) / "phase1_seed_summary.parquet")
        result.to_csv(Path(output_root) / "phase1_seed_summary.csv", index=False)
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
        for required in (
            run_dir / "artifacts" / "latents_final.safetensors",
            run_dir / "analysis" / "geometry_final.json",
            run_dir / "analysis" / "invariance_final.json",
        ):
            if not required.exists():
                errors.append(f"missing artifact: {required}")
        completed_evaluations = [
            candidate
            for candidate in (run_dir / "evaluations").glob("robustness-*-attempt*")
            if (candidate / "COMPLETED").exists()
            and (candidate / "input_pgd.parquet").exists()
            and (candidate / "latent_pgd.parquet").exists()
        ]
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
        diagnostics = [
            candidate
            for candidate in (run_dir / "evaluations").glob("attack-diagnostics-*-attempt*")
            if (candidate / "COMPLETED").exists() and (candidate / "diagnostics.json").exists()
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
