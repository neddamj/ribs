"""Natural and adversarial semantic-representation collision analyses."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from .attacks import _seeded_rng


def nearest_opposing(
    latents: Tensor, labels: Tensor, block_size: int = 512
) -> tuple[Tensor, Tensor]:
    z = latents.detach().cpu().double()
    labels = labels.detach().cpu()
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    nearest_distance = torch.full((len(z),), float("inf"), dtype=torch.float64)
    nearest_index = torch.full((len(z),), -1, dtype=torch.long)
    for start in range(0, len(z), block_size):
        end = min(start + block_size, len(z))
        distance = torch.cdist(z[start:end], z)
        opposing = labels[start:end, None] != labels[None, :]
        values, indices = distance.masked_fill(~opposing, float("inf")).min(dim=1)
        nearest_distance[start:end] = values.cpu()
        nearest_index[start:end] = indices.cpu()
    return nearest_distance, nearest_index


def calibrate_collision_threshold(
    tuning_latents: Tensor, tuning_labels: Tensor, percentile: float = 5.0, block_size: int = 512
) -> float:
    z = tuning_latents.detach().cpu().double()
    tuning_labels = tuning_labels.detach().cpu()
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    nearest_same = torch.full((len(z),), float("inf"), dtype=torch.float64)
    for start in range(0, len(z), block_size):
        end = min(start + block_size, len(z))
        distances = torch.cdist(z[start:end], z)
        same = tuning_labels[start:end, None] == tuning_labels[None, :]
        same[:, start:end].fill_diagonal_(False)
        nearest_same[start:end] = distances.masked_fill(~same, float("inf")).min(dim=1).values.cpu()
    return float(torch.quantile(nearest_same[torch.isfinite(nearest_same)], percentile / 100.0))


def natural_collision_metrics(
    latents: Tensor, labels: Tensor, threshold: float | None = None
) -> dict[str, float]:
    distances, _ = nearest_opposing(latents, labels)
    result = {
        "nearest_opposing_mean": float(distances.mean()),
        "nearest_opposing_median": float(distances.median()),
        "nearest_opposing_p05": float(torch.quantile(distances, 0.05)),
        "nearest_opposing_p95": float(torch.quantile(distances, 0.95)),
    }
    if threshold is not None:
        result["continuous_collision_rate"] = float((distances < threshold).float().mean())
        result["collision_threshold"] = threshold
    return result


def _project_linf(x: Tensor, source: Tensor, epsilon: float) -> Tensor:
    return torch.max(torch.min(x, source + epsilon), source - epsilon).clamp(0.0, 1.0)


def targeted_collision_attack(
    model: torch.nn.Module,
    reference_classifier: torch.nn.Module,
    source: Tensor,
    source_label: Tensor,
    target: Tensor,
    epsilon: float,
    threshold: float,
    steps: int = 200,
    restarts: int = 5,
    lambda_sem: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    model.eval()
    reference_classifier.eval()
    with torch.no_grad():
        target_output = model(target, sample=False)
        target_z = target_output.canonical_latent.detach().flatten(1)
        target_normalized = F.normalize(target_z.float(), dim=1, eps=1e-12)
        is_vq = "code_indices" in target_output.metadata
        is_quantized = (
            "quantized_latent" in target_output.metadata
            and getattr(model, "bits", None) is not None
            and int(model.bits) < 32
        )
        if is_vq:
            target_codes = target_output.metadata["code_indices"].detach()
            target_pre_goal = model.codebook[target_codes].detach()
        elif is_quantized:
            target_pre_goal = target_output.metadata["quantized_latent"].detach()
        else:
            target_pre_goal = None
    best = source.detach().clone()
    best_distance = torch.full((source.shape[0],), float("inf"), device=source.device)
    best_score = torch.full((source.shape[0],), float("inf"), device=source.device)
    best_success = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
    attack_steps = max(0, int(steps)) if epsilon > 0 else 0
    attack_restarts = max(1, int(restarts)) if epsilon > 0 else 1
    with _seeded_rng(source, seed):
        generator = torch.Generator(device=source.device).manual_seed(seed + 1)
        for restart in range(attack_restarts):
            if restart == 0:
                adv = source.detach().clone()
            else:
                adv = _project_linf(
                    source
                    + torch.empty_like(source).uniform_(-epsilon, epsilon, generator=generator),
                    source,
                    epsilon,
                )
            for step_index in range(attack_steps + 1):
                adv.requires_grad_(True)
                output = model(adv, sample=False)
                normalized = F.normalize(
                    output.canonical_latent.float().flatten(1), dim=1, eps=1e-12
                )
                normalized_distance = (normalized - target_normalized).norm(dim=1)
                if is_vq:
                    pre = output.pre_bottleneck
                    attack_distance = (pre - target_pre_goal).flatten(1).norm(dim=1)
                    collision_condition = (output.metadata["code_indices"] == target_codes).all(
                        dim=1
                    )
                elif is_quantized:
                    pre = output.pre_bottleneck
                    attack_distance = (pre - target_pre_goal).flatten(1).norm(dim=1)
                    collision_condition = (
                        (output.metadata["quantized_latent"] == target_pre_goal)
                        .flatten(1)
                        .all(dim=1)
                    )
                else:
                    attack_distance = normalized_distance
                    collision_condition = normalized_distance < threshold
                source_loss = F.cross_entropy(
                    reference_classifier(adv).logits.float(), source_label, reduction="none"
                )
                objective = attack_distance + lambda_sem * source_loss
                with torch.no_grad():
                    reference_prediction = reference_classifier(adv).logits.argmax(dim=-1)
                    success = (reference_prediction == source_label) & collision_condition
                    replace = (success & ~best_success) | (
                        (success == best_success) & (attack_distance < best_score)
                    )
                    best = torch.where(replace.view(-1, 1, 1, 1), adv, best)
                    best_distance = torch.where(replace, normalized_distance, best_distance)
                    best_score = torch.where(replace, attack_distance, best_score)
                    best_success = torch.where(replace, success, best_success)
                if step_index == attack_steps:
                    break
                grad = torch.autograd.grad(objective.sum(), adv)[0]
                with torch.no_grad():
                    direction = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(
                        -1, 1, 1, 1
                    )
                    adv = _project_linf(
                        adv.detach() - (2 * epsilon / max(1, attack_steps)) * direction,
                        source,
                        epsilon,
                    )
    return {
        "adversarial": best,
        "distance": best_distance,
        "successful": best_success,
        "criterion": "exact_vq" if is_vq else "exact_quantized" if is_quantized else "threshold",
    }


def collision_records(
    sample_ids: list[str], labels: Tensor, distances: Tensor, indices: Tensor, threshold: float
) -> pd.DataFrame:
    labels_cpu = labels.cpu().tolist()
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels_cpu,
            "opposing_sample_id": [sample_ids[int(index)] for index in indices],
            "opposing_label": [labels_cpu[int(index)] for index in indices],
            "distance": distances.tolist(),
            "threshold": threshold,
            "collision": (distances < threshold).tolist(),
        }
    )
