"""Input- and latent-space projected-gradient attacks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AttackResult:
    adversarial: Tensor
    loss: Tensor
    successful: Tensor
    initial_loss: Tensor
    restart: Tensor | None = None


@contextmanager
def _seeded_rng(reference: Tensor, seed: int):
    """Use a local RNG stream, including stochastic model forwards."""
    devices = []
    if reference.device.type == "cuda":
        devices = [reference.device.index if reference.device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if reference.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def _select(
    best_x: Tensor,
    best_loss: Tensor,
    best_success: Tensor,
    candidate_x: Tensor,
    candidate_loss: Tensor,
    candidate_success: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    replace = (candidate_success & ~best_success) | (candidate_success == best_success) & (
        candidate_loss > best_loss
    )
    return (
        torch.where(replace.view(-1, *([1] * (best_x.ndim - 1))), candidate_x, best_x),
        torch.where(replace, candidate_loss, best_loss),
        torch.where(replace, candidate_success, best_success),
    )


def _eot_loss(model: torch.nn.Module, x: Tensor, y: Tensor, samples: int) -> tuple[Tensor, Tensor]:
    """Return expected CE and mean class probabilities for EoT prediction."""
    losses = []
    probabilities = None
    for _ in range(max(1, samples)):
        output = model(x, sample=samples > 1)
        current = torch.softmax(output.logits.float(), dim=-1)
        probabilities = current if probabilities is None else probabilities + current
        losses.append(F.cross_entropy(output.logits.float(), y, reduction="none"))
    return torch.stack(losses).mean(0), probabilities / max(1, samples)


def input_pgd(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    epsilon: float,
    steps: int = 40,
    restarts: int = 5,
    eot_samples: int = 1,
    seed: int = 0,
) -> AttackResult:
    model.eval()
    x = x.detach().float()
    with _seeded_rng(x, seed):
        with torch.no_grad():
            initial_loss, initial_probabilities = _eot_loss(model, x, y, eot_samples)
            clean_pred = initial_probabilities.argmax(dim=-1)
        initial_restart = torch.full_like(y, -1, dtype=torch.long)
        if epsilon == 0.0:
            return AttackResult(
                x, initial_loss.detach(), clean_pred.ne(y), initial_loss.detach(), initial_restart
            )
        generator = torch.Generator(device=x.device).manual_seed(seed + 1)
        best_x = x.clone()
        best_loss = initial_loss.detach().clone()
        best_success = clean_pred.ne(y)
        best_restart = initial_restart
        step_size = 2.0 * epsilon / max(1, steps)
        for restart in range(restarts):
            adv = (
                (x + torch.empty_like(x).uniform_(-epsilon, epsilon, generator=generator))
                .clamp(0.0, 1.0)
                .detach()
            )
            for _ in range(steps + 1):
                adv.requires_grad_(True)
                losses, probabilities = _eot_loss(model, adv, y, eot_samples)
                grad = torch.autograd.grad(losses.sum(), adv)[0]
                success = probabilities.detach().argmax(dim=-1).ne(y)
                with torch.no_grad():
                    replace = (success & ~best_success) | (success == best_success) & (
                        losses > best_loss
                    )
                    best_x, best_loss, best_success = _select(
                        best_x, best_loss, best_success, adv.detach(), losses.detach(), success
                    )
                    best_restart = torch.where(
                        replace, torch.full_like(best_restart, restart), best_restart
                    )
                    adv = (adv.detach() + step_size * grad.sign()).clamp(0.0, 1.0)
                    adv = torch.max(torch.min(adv, x + epsilon), x - epsilon).clamp(0.0, 1.0)
        if torch.any(best_loss + 1e-6 < initial_loss):
            raise RuntimeError("PGD retained loss is below its clean initial loss")
        if (best_x - x).abs().flatten(1).amax(1).max() > epsilon + 1e-6:
            raise RuntimeError("PGD produced an example outside the L-infinity constraint")
        if best_x.min() < -1e-6 or best_x.max() > 1.0 + 1e-6:
            raise RuntimeError("PGD produced pixels outside [0, 1]")
        return AttackResult(best_x, best_loss, best_success, initial_loss.detach(), best_restart)


def project_l2(delta: Tensor, radius: Tensor) -> Tensor:
    norm = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(norm), radius / norm)
    return delta * factor.view(-1, 1, *([1] * (delta.ndim - 2)))


def latent_pgd(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    rho: float,
    steps: int = 40,
    restarts: int = 5,
    seed: int = 0,
) -> AttackResult:
    model.eval()
    with torch.no_grad():
        output = model(x, sample=False)
        z = output.latent.detach()
        initial_logits = model.classify_latent(z)
        initial_loss = F.cross_entropy(initial_logits.float(), y, reduction="none")
        clean_pred = initial_logits.argmax(dim=-1)
    radius = rho * z.flatten(1).norm(dim=1).clamp_min(1e-12)
    generator = torch.Generator(device=z.device).manual_seed(seed)
    best_z = z.clone()
    best_loss = initial_loss.clone()
    best_success = clean_pred.ne(y)
    step = 2.0 * radius / max(1, steps)
    for _ in range(restarts):
        noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        dimension = max(1, noise[0].numel())
        radial = torch.rand(z.shape[0], device=z.device, generator=generator).pow(1.0 / dimension)
        noise = project_l2(noise, radius * radial)
        adv = z + noise
        for _ in range(steps + 1):
            adv.requires_grad_(True)
            logits = model.classify_latent(adv)
            losses = F.cross_entropy(logits.float(), y, reduction="none")
            grad = torch.autograd.grad(losses.sum(), adv)[0]
            direction = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(
                -1, *([1] * (grad.ndim - 1))
            )
            success = logits.detach().argmax(dim=-1).ne(y)
            with torch.no_grad():
                best_z, best_loss, best_success = _select(
                    best_z, best_loss, best_success, adv.detach(), losses.detach(), success
                )
                adv = adv.detach() + direction * step.view(-1, *([1] * (adv.ndim - 1)))
                adv = z + project_l2(adv - z, radius)
    if torch.any(best_loss + 1e-6 < initial_loss):
        raise RuntimeError("Latent PGD retained loss is below its clean initial loss")
    return AttackResult(best_z, best_loss, best_success, initial_loss)


def prequantization_latent_pgd(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    rho: float,
    steps: int = 40,
    restarts: int = 5,
    seed: int = 0,
) -> AttackResult:
    """Attack a continuous encoder output and reapply exact quantization.

    Models in the discrete families expose ``classify_pre_bottleneck`` with
    an STE/BPDA derivative. The forward value remains the exact quantized
    classifier input, so this is the primary discrete latent attack surface.
    """
    if not hasattr(model, "classify_pre_bottleneck"):
        raise TypeError("Model does not expose a pre-quantization classifier")
    model.eval()
    with torch.no_grad():
        output = model(x, sample=False)
        pre = output.pre_bottleneck.detach().flatten(1)
        initial_logits = model.classify_pre_bottleneck(pre)
        initial_loss = F.cross_entropy(initial_logits.float(), y, reduction="none")
        clean_pred = initial_logits.argmax(dim=-1)
    radius = rho * pre.norm(dim=1).clamp_min(1e-12)
    generator = torch.Generator(device=pre.device).manual_seed(seed)
    best = pre.clone()
    best_loss = initial_loss.clone()
    best_success = clean_pred.ne(y)
    step = 2.0 * radius / max(1, steps)
    for _ in range(restarts):
        noise = torch.randn(pre.shape, generator=generator, device=pre.device, dtype=pre.dtype)
        dimension = max(1, noise[0].numel())
        radial = torch.rand(pre.shape[0], device=pre.device, generator=generator).pow(
            1.0 / dimension
        )
        noise = project_l2(noise, radius * radial)
        adv = pre + noise
        for _ in range(steps + 1):
            adv.requires_grad_(True)
            logits = model.classify_pre_bottleneck(adv)
            losses = F.cross_entropy(logits.float(), y, reduction="none")
            grad = torch.autograd.grad(losses.sum(), adv)[0]
            direction = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
            success = logits.detach().argmax(dim=-1).ne(y)
            with torch.no_grad():
                best, best_loss, best_success = _select(
                    best, best_loss, best_success, adv.detach(), losses.detach(), success
                )
                adv = adv.detach() + direction * step.view(-1, 1)
                adv = pre + project_l2(adv - pre, radius)
    if torch.any(best_loss + 1e-6 < initial_loss):
        raise RuntimeError("Pre-quantization PGD retained loss is below its initial loss")
    return AttackResult(best, best_loss, best_success, initial_loss)
