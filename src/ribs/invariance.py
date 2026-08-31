"""Fixed nuisance-transform invariance measurements."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import functional as TF


def _translate(images: Tensor, dx: int, dy: int) -> Tensor:
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    x0, y0 = 4 - dx, 4 - dy
    return padded[:, :, y0 : y0 + images.shape[2], x0 : x0 + images.shape[3]]


def transform_variants(
    images: Tensor, noise_seed: int = 0, sample_ids: list[str] | None = None
) -> dict[str, Tensor]:
    if sample_ids is None:
        sample_ids = [str(index) for index in range(len(images))]
    if len(sample_ids) != len(images):
        raise ValueError("sample_ids must align with images")
    noise_variants = []
    for suffix in ("a", "b"):
        samples = []
        for image, sample_id in zip(images, sample_ids):
            digest = hashlib.sha256(f"{noise_seed}\0{sample_id}\0{suffix}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device=images.device).manual_seed(seed)
            samples.append(
                torch.randn(image.shape, device=images.device, generator=generator) * 0.02
            )
        noise_variants.append(torch.stack(samples))
    noise_a, noise_b = noise_variants
    return {
        "translation_left": _translate(images, -4, 0),
        "translation_right": _translate(images, 4, 0),
        "translation_up": _translate(images, 0, -4),
        "translation_down": _translate(images, 0, 4),
        "brightness_low": TF.adjust_brightness(images, 0.9).clamp(0, 1),
        "brightness_high": TF.adjust_brightness(images, 1.1).clamp(0, 1),
        "contrast_low": TF.adjust_contrast(images, 0.9).clamp(0, 1),
        "contrast_high": TF.adjust_contrast(images, 1.1).clamp(0, 1),
        "noise_a": (images + noise_a).clamp(0, 1),
        "noise_b": (images + noise_b).clamp(0, 1),
        "horizontal_flip": TF.hflip(images),
    }


@torch.no_grad()
def invariance_metrics(
    model: torch.nn.Module,
    images: Tensor,
    labels: Tensor | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, float]:
    model.eval()
    base = model(images, sample=False)
    base_latent = F.normalize(base.canonical_latent.float().flatten(1), dim=1, eps=1e-12)
    base_prediction = base.logits.argmax(dim=-1)
    groups: dict[str, list[float]] = defaultdict(list)
    consistency: dict[str, list[float]] = defaultdict(list)
    accuracy: dict[str, list[float]] = defaultdict(list)
    result = {}
    for name, variant in transform_variants(images, sample_ids=sample_ids).items():
        output = model(variant, sample=False)
        variant_latent = F.normalize(output.canonical_latent.float().flatten(1), dim=1, eps=1e-12)
        distance = (base_latent - variant_latent).norm(dim=1)
        variant_prediction = output.logits.argmax(dim=-1)
        group = name.split("_")[0] if name != "horizontal_flip" else "flip"
        if name.startswith("noise"):
            group = "noise"
        groups[group].extend(distance.cpu().tolist())
        consistency[group].extend((variant_prediction == base_prediction).float().cpu().tolist())
        if labels is not None:
            accuracy[group].extend((variant_prediction == labels).float().cpu().tolist())
        result[f"nuisance_distance_{name}"] = float(distance.mean())
        result[f"prediction_consistency_{name}"] = float(
            (variant_prediction == base_prediction).float().mean()
        )
        if labels is not None:
            result[f"accuracy_{name}"] = float((variant_prediction == labels).float().mean())
    for group in ("translation", "brightness", "contrast", "noise", "flip"):
        result[f"nuisance_distance_{group}"] = sum(groups[group]) / max(1, len(groups[group]))
        result[f"prediction_consistency_{group}"] = sum(consistency[group]) / max(
            1, len(consistency[group])
        )
        if labels is not None:
            result[f"accuracy_{group}"] = sum(accuracy[group]) / max(1, len(accuracy[group]))
    transform_groups = ("translation", "brightness", "contrast", "noise", "flip")
    result["nuisance_distance_macro"] = sum(
        result[f"nuisance_distance_{group}"] for group in transform_groups
    ) / len(transform_groups)
    if labels is not None:
        result["base_accuracy"] = float((base_prediction == labels).float().mean())
    return result
