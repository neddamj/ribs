"""Blockwise representation-geometry estimators."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import Tensor


def normalize_latents(latents: Tensor) -> tuple[Tensor, int]:
    norms = latents.norm(dim=1, keepdim=True)
    near_zero = int((norms.squeeze(1) < 1e-12).sum())
    return latents / norms.clamp_min(1e-12), near_zero


def pairwise_class_distances(
    latents: Tensor, labels: Tensor, block_size: int = 512
) -> tuple[float, float, Tensor]:
    details = pairwise_class_distance_details(latents, labels, block_size)
    return details["intra"], details["inter"], details["nearest"]


def pairwise_class_distance_details(
    latents: Tensor, labels: Tensor, block_size: int = 512
) -> dict[str, Any]:
    """Compute equal-anchor-weighted distances without materializing an NxN matrix."""
    normalized, _ = normalize_latents(latents.detach().cpu().double())
    labels = labels.cpu()
    n = normalized.shape[0]
    intra_by_anchor = torch.full((n,), float("nan"), dtype=torch.float64)
    inter_by_anchor = torch.full((n,), float("nan"), dtype=torch.float64)
    nearest = torch.full((n,), float("inf"), dtype=torch.float64)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        distances = torch.cdist(normalized[start:end], normalized, p=2)
        same = labels[start:end, None] == labels[None, :]
        same[:, start:end].fill_diagonal_(False)
        different = labels[start:end, None] != labels[None, :]
        same_count = same.sum(dim=1)
        different_count = different.sum(dim=1)
        intra_values = distances.masked_fill(~same, 0.0).sum(dim=1) / same_count.clamp_min(1)
        intra_values = intra_values.masked_fill(same_count == 0, float("nan"))
        inter_values = distances.masked_fill(~different, 0.0).sum(
            dim=1
        ) / different_count.clamp_min(1)
        inter_values = inter_values.masked_fill(different_count == 0, float("nan"))
        intra_by_anchor[start:end] = intra_values.cpu()
        inter_by_anchor[start:end] = inter_values.cpu()
        nearest[start:end] = distances.masked_fill(~different, float("inf")).min(dim=1).values.cpu()
    per_class = {}
    for class_label in torch.unique(labels, sorted=True):
        mask = labels == class_label
        per_class[str(int(class_label))] = {
            "intra_class_distance": float(torch.nanmean(intra_by_anchor[mask])),
            "inter_class_distance": float(torch.nanmean(inter_by_anchor[mask])),
            "nearest_opposing_distance": float(torch.nanmean(nearest[mask])),
        }
    macro = {
        metric: float(np.nanmean([values[metric] for values in per_class.values()]))
        for metric in (
            "intra_class_distance",
            "inter_class_distance",
            "nearest_opposing_distance",
        )
    }
    return {
        "intra": float(torch.nanmean(intra_by_anchor)),
        "inter": float(torch.nanmean(inter_by_anchor)),
        "nearest": nearest,
        "per_class": per_class,
        "macro": macro,
        "intra_by_anchor": intra_by_anchor,
        "inter_by_anchor": inter_by_anchor,
    }


def effective_rank(latents: Tensor) -> tuple[float, Tensor]:
    values = latents.detach().cpu().double()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    threshold = eigenvalues.max().item() * 1e-12 if eigenvalues.numel() else 0.0
    eigenvalues = eigenvalues[eigenvalues > threshold]
    if not eigenvalues.numel():
        return 0.0, eigenvalues
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return float(rank), eigenvalues


def geometry_metrics(latents: Tensor, labels: Tensor, block_size: int = 512) -> dict[str, Any]:
    _, near_zero = normalize_latents(latents.double())
    details = pairwise_class_distance_details(latents, labels, block_size)
    intra, inter, nearest = details["intra"], details["inter"], details["nearest"]
    rank, eigenvalues = effective_rank(latents)
    return {
        "intra_class_distance": intra,
        "inter_class_distance": inter,
        "separation_ratio": inter / (intra + 1e-12),
        "median_nearest_opposing_distance": float(nearest.median()),
        "mean_nearest_opposing_distance": float(nearest.mean()),
        "effective_rank": rank,
        "nominal_dimension": int(latents.shape[1]),
        "near_zero_latents": near_zero,
        "eigenvalues": eigenvalues.cpu().tolist(),
        "per_class": details["per_class"],
        "macro_intra_class_distance": details["macro"]["intra_class_distance"],
        "macro_inter_class_distance": details["macro"]["inter_class_distance"],
        "macro_nearest_opposing_distance": details["macro"]["nearest_opposing_distance"],
    }


def deterministic_partners(
    sample_ids: list[str], labels: Tensor, per_class: int = 10
) -> list[tuple[int, int, bool]]:
    pairs: list[tuple[int, int, bool]] = []
    ids = np.asarray(sample_ids)
    labels_np = labels.cpu().numpy()
    for i, sample_id in enumerate(ids):
        same = np.flatnonzero(labels_np == labels_np[i])
        different = np.flatnonzero(labels_np != labels_np[i])
        same = same[same != i]
        for pool, is_same in ((same, True), (different, False)):
            order = sorted(
                pool.tolist(),
                key=lambda j: (
                    hashlib.sha256(f"{sample_id}\0{ids[j]}".encode()).digest(),
                    ids[j],
                ),
            )[:per_class]
            pairs.extend((i, j, is_same) for j in order)
    return pairs


def contraction_metrics(
    images: Tensor, latents: Tensor, labels: Tensor, sample_ids: list[str], per_anchor: int = 10
) -> dict[str, float]:
    pairs = deterministic_partners(sample_ids, labels, per_anchor)
    raw_latents = latents.double()
    normalized_latents, _ = normalize_latents(raw_latents)
    values_by_kind = {
        "same": {"raw": [], "normalized": []},
        "different": {"raw": [], "normalized": []},
    }
    for i, j, same in pairs:
        input_distance = (images[i].float() - images[j].float()).flatten().norm().item() / images[
            i
        ].numel() ** 0.5
        raw_distance = (raw_latents[i] - raw_latents[j]).norm().item() / raw_latents.shape[1] ** 0.5
        normalized_distance = (
            normalized_latents[i] - normalized_latents[j]
        ).norm().item() / raw_latents.shape[1] ** 0.5
        kind = "same" if same else "different"
        values_by_kind[kind]["raw"].append(input_distance / max(raw_distance, 1e-12))
        values_by_kind[kind]["normalized"].append(input_distance / max(normalized_distance, 1e-12))
    result = {}
    for kind, variants in values_by_kind.items():
        for variant, values in variants.items():
            suffix = "" if variant == "raw" else "_normalized"
            result[f"contraction_{kind}_median{suffix}"] = (
                float(np.median(values)) if values else float("nan")
            )
            result[f"contraction_{kind}_p95{suffix}"] = (
                float(np.percentile(values, 95)) if values else float("nan")
            )
    return result


def encoder_spectral_norm(
    model: torch.nn.Module, images: Tensor, iterations: int = 20, seed: int = 0
) -> Tensor:
    """Estimate one encoder Jacobian spectral norm per image with JVP/VJP."""
    model.eval()
    estimates = []

    def encoder_value(value: Tensor) -> Tensor:
        output = model(value, sample=False)
        if hasattr(model, "classify_pre_bottleneck"):
            return output.pre_bottleneck.flatten(1)
        return output.canonical_latent.flatten(1)

    for image_index, image in enumerate(images):
        x = image.detach().clone().unsqueeze(0).requires_grad_(True)
        generator = torch.Generator(device=x.device).manual_seed(seed + image_index)
        vector = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        vector = vector / vector.flatten().norm().clamp_min(1e-12)
        sigma = torch.tensor(0.0, device=x.device)
        for _ in range(iterations):
            _, jvp = torch.autograd.functional.jvp(encoder_value, x, vector, create_graph=True)
            sigma = jvp.flatten().norm()
            unit_output = jvp / sigma.clamp_min(1e-12)
            features = encoder_value(x)
            vector = torch.autograd.grad(
                features, x, grad_outputs=unit_output, retain_graph=False, create_graph=False
            )[0]
            vector = vector / vector.flatten().norm().clamp_min(1e-12)
        estimates.append(sigma.detach())
    return torch.stack(estimates) if estimates else torch.empty(0)
