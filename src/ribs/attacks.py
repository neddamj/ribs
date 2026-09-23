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
    # Highest CE encountered, retained independently of the success-priority
    # candidate used for the attack prediction.
    retained_loss: Tensor | None = None


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


_EOT_CHUNK_SIZE = 2


def _eot_loss(model: torch.nn.Module, x: Tensor, y: Tensor, samples: int) -> tuple[Tensor, Tensor]:
    """Return expected CE and mean class probabilities for EoT prediction."""
    count = max(1, samples)
    loss_sum = torch.zeros(len(x), device=x.device)
    probability_sum = None
    for start in range(0, count, _EOT_CHUNK_SIZE):
        chunk = min(_EOT_CHUNK_SIZE, count - start)
        expanded = x.repeat_interleave(chunk, dim=0)
        expanded_labels = y.repeat_interleave(chunk)
        logits = model(expanded, sample=count > 1).logits.float()
        current_loss = F.cross_entropy(logits, expanded_labels, reduction="none").view(
            len(x), chunk
        )
        current_probability = torch.softmax(logits, dim=-1).view(len(x), chunk, -1)
        loss_sum = loss_sum + current_loss.sum(dim=1)
        chunk_probability = current_probability.sum(dim=1)
        probability_sum = (
            chunk_probability if probability_sum is None else probability_sum + chunk_probability
        )
    return loss_sum / count, probability_sum / count


def _eot_loss_with_gradient(
    model: torch.nn.Module, x: Tensor, y: Tensor, samples: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute EoT values and gradient without retaining all sample graphs.

    The prescribed EoT count is unchanged. Each stochastic forward is
    differentiated and released before the next one, so memory scales with a
    single sample graph rather than with the number of EoT samples.
    """
    count = max(1, samples)
    loss_sum = torch.zeros(len(x), device=x.device)
    probability_sum = None
    gradient = torch.zeros_like(x)
    for start in range(0, count, _EOT_CHUNK_SIZE):
        chunk = min(_EOT_CHUNK_SIZE, count - start)
        expanded = x.repeat_interleave(chunk, dim=0)
        expanded_labels = y.repeat_interleave(chunk)
        logits = model(expanded, sample=count > 1).logits.float()
        current_loss = F.cross_entropy(logits, expanded_labels, reduction="none").view(
            len(x), chunk
        )
        gradient = gradient + torch.autograd.grad(current_loss.sum(), x)[0]
        loss_sum = loss_sum + current_loss.detach().sum(dim=1)
        chunk_probability = (
            torch.softmax(logits.detach(), dim=-1).view(len(x), chunk, -1).sum(dim=1)
        )
        probability_sum = (
            chunk_probability if probability_sum is None else probability_sum + chunk_probability
        )
    return (
        loss_sum / count,
        probability_sum / count,
        gradient / count,
    )


def input_pgd(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    epsilon: float,
    steps: int = 40,
    restarts: int = 5,
    eot_samples: int = 1,
    seed: int = 0,
    initial_adversarial: Tensor | None = None,
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
                x,
                initial_loss.detach(),
                clean_pred.ne(y),
                initial_loss.detach(),
                initial_restart,
                initial_loss.detach(),
            )
        generator = torch.Generator(device=x.device).manual_seed(seed + 1)
        best_x = x.clone()
        best_loss = initial_loss.detach().clone()
        best_success = clean_pred.ne(y)
        best_restart = initial_restart
        retained_loss = initial_loss.detach().clone()
        if initial_adversarial is not None:
            candidate = initial_adversarial.detach().float()
            if candidate.shape != x.shape:
                raise ValueError("Initial input candidate has the wrong shape")
            if (candidate - x).abs().flatten(1).amax(1).max() > epsilon + 1e-6:
                raise ValueError("Initial input candidate is outside the attack constraint")
            with torch.no_grad():
                candidate_loss, candidate_probabilities = _eot_loss(
                    model, candidate, y, eot_samples
                )
                retained_loss = torch.maximum(retained_loss, candidate_loss.detach())
                candidate_success = candidate_probabilities.argmax(dim=-1).ne(y)
                replace = (candidate_success & ~best_success) | (
                    candidate_success == best_success
                ) & (candidate_loss > best_loss)
                best_x, best_loss, best_success = _select(
                    best_x,
                    best_loss,
                    best_success,
                    candidate,
                    candidate_loss,
                    candidate_success,
                )
                best_restart = torch.where(replace, torch.full_like(best_restart, -2), best_restart)
            # Nested robustness only needs one valid adversarial example. If
            # every sample is already broken by the carried smaller-radius
            # candidate, additional random starts cannot change the curve.
            if bool(best_success.all()):
                return AttackResult(
                    best_x,
                    best_loss,
                    best_success,
                    initial_loss.detach(),
                    best_restart,
                    retained_loss,
                )
        attack_steps = max(1, steps)
        step_size = 2.0 * epsilon / attack_steps
        for restart in range(restarts):
            adv = (
                (x + torch.empty_like(x).uniform_(-epsilon, epsilon, generator=generator))
                .clamp(0.0, 1.0)
                .detach()
            )
            # Evaluate the initialization and every updated point, including
            # the candidate produced by the final gradient step.
            for step_index in range(attack_steps + 1):
                adv.requires_grad_(True)
                losses, probabilities, grad = _eot_loss_with_gradient(model, adv, y, eot_samples)
                success = probabilities.detach().argmax(dim=-1).ne(y)
                with torch.no_grad():
                    retained_loss = torch.maximum(retained_loss, losses.detach())
                    replace = (success & ~best_success) | (success == best_success) & (
                        losses > best_loss
                    )
                    best_x, best_loss, best_success = _select(
                        best_x, best_loss, best_success, adv.detach(), losses.detach(), success
                    )
                    best_restart = torch.where(
                        replace, torch.full_like(best_restart, restart), best_restart
                    )
                if step_index == attack_steps:
                    break
                with torch.no_grad():
                    adv = (adv.detach() + step_size * grad.sign()).clamp(0.0, 1.0)
                    adv = torch.max(torch.min(adv, x + epsilon), x - epsilon).clamp(0.0, 1.0)
        if torch.any(retained_loss + 1e-6 < initial_loss):
            raise RuntimeError("PGD highest retained loss is below its clean initial loss")
        if (best_x - x).abs().flatten(1).amax(1).max() > epsilon + 1e-6:
            raise RuntimeError("PGD produced an example outside the L-infinity constraint")
        if best_x.min() < -1e-6 or best_x.max() > 1.0 + 1e-6:
            raise RuntimeError("PGD produced pixels outside [0, 1]")
        return AttackResult(
            best_x,
            best_loss,
            best_success,
            initial_loss.detach(),
            best_restart,
            retained_loss,
        )


def project_l2(delta: Tensor, radius: Tensor) -> Tensor:
    norm = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(norm), radius / norm)
    return delta * factor.view(-1, 1, *([1] * (delta.ndim - 2)))


def _uniform_l2_noise(reference: Tensor, radius: Tensor, generator: torch.Generator) -> Tensor:
    """Draw independently and uniformly from each sample's L2 ball."""
    noise = torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )
    flat_norm = noise.flatten(1).norm(dim=1).clamp_min(1e-12)
    direction = noise / flat_norm.view(-1, *([1] * (noise.ndim - 1)))
    dimension = max(1, noise[0].numel())
    radial = torch.rand(reference.shape[0], device=reference.device, generator=generator).pow(
        1.0 / dimension
    )
    scale = radius * radial
    return direction * scale.view(-1, *([1] * (noise.ndim - 1)))


def latent_pgd(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    rho: float,
    steps: int = 40,
    restarts: int = 5,
    seed: int = 0,
    initial_adversarial: Tensor | None = None,
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
    best_restart = torch.full_like(y, -1, dtype=torch.long)
    retained_loss = initial_loss.clone()
    if initial_adversarial is not None:
        candidate = initial_adversarial.detach()
        if candidate.shape != z.shape:
            raise ValueError("Initial latent candidate has the wrong shape")
        if torch.any((candidate - z).flatten(1).norm(dim=1) > radius + 1e-6):
            raise ValueError("Initial latent candidate is outside the attack constraint")
        with torch.no_grad():
            candidate_logits = model.classify_latent(candidate)
            candidate_loss = F.cross_entropy(candidate_logits.float(), y, reduction="none")
            retained_loss = torch.maximum(retained_loss, candidate_loss)
            candidate_success = candidate_logits.argmax(dim=-1).ne(y)
            replace = (candidate_success & ~best_success) | (candidate_success == best_success) & (
                candidate_loss > best_loss
            )
            best_z, best_loss, best_success = _select(
                best_z,
                best_loss,
                best_success,
                candidate,
                candidate_loss,
                candidate_success,
            )
            best_restart = torch.where(replace, torch.full_like(best_restart, -2), best_restart)
    attack_steps = max(1, steps)
    for restart in range(restarts):
        noise = _uniform_l2_noise(z, radius, generator)
        adv = z + noise
        for step_index in range(attack_steps + 1):
            adv.requires_grad_(True)
            logits = model.classify_latent(adv)
            losses = F.cross_entropy(logits.float(), y, reduction="none")
            success = logits.detach().argmax(dim=-1).ne(y)
            with torch.no_grad():
                retained_loss = torch.maximum(retained_loss, losses.detach())
                replace = (success & ~best_success) | (success == best_success) & (
                    losses > best_loss
                )
                best_z, best_loss, best_success = _select(
                    best_z, best_loss, best_success, adv.detach(), losses.detach(), success
                )
                best_restart = torch.where(
                    replace, torch.full_like(best_restart, restart), best_restart
                )
            if step_index == attack_steps:
                break
            grad = torch.autograd.grad(losses.sum(), adv)[0]
            direction = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(
                -1, *([1] * (grad.ndim - 1))
            )
            with torch.no_grad():
                adv = adv.detach() + direction * step.view(-1, *([1] * (adv.ndim - 1)))
                adv = z + project_l2(adv - z, radius)
    if torch.any(retained_loss + 1e-6 < initial_loss):
        raise RuntimeError("Latent PGD highest retained loss is below its clean initial loss")
    return AttackResult(best_z, best_loss, best_success, initial_loss, best_restart, retained_loss)


def prequantization_latent_pgd(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    rho: float,
    steps: int = 40,
    restarts: int = 5,
    seed: int = 0,
    initial_adversarial: Tensor | None = None,
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
    best_restart = torch.full_like(y, -1, dtype=torch.long)
    retained_loss = initial_loss.clone()
    if initial_adversarial is not None:
        candidate = initial_adversarial.detach().flatten(1)
        if candidate.shape != pre.shape:
            raise ValueError("Initial pre-quantization candidate has the wrong shape")
        if torch.any((candidate - pre).norm(dim=1) > radius + 1e-6):
            raise ValueError("Initial pre-quantization candidate is outside the attack constraint")
        with torch.no_grad():
            candidate_logits = model.classify_pre_bottleneck(candidate)
            candidate_loss = F.cross_entropy(candidate_logits.float(), y, reduction="none")
            retained_loss = torch.maximum(retained_loss, candidate_loss)
            candidate_success = candidate_logits.argmax(dim=-1).ne(y)
            replace = (candidate_success & ~best_success) | (candidate_success == best_success) & (
                candidate_loss > best_loss
            )
            best, best_loss, best_success = _select(
                best,
                best_loss,
                best_success,
                candidate,
                candidate_loss,
                candidate_success,
            )
            best_restart = torch.where(replace, torch.full_like(best_restart, -2), best_restart)
    attack_steps = max(1, steps)
    for restart in range(restarts):
        noise = _uniform_l2_noise(pre, radius, generator)
        adv = pre + noise
        for step_index in range(attack_steps + 1):
            adv.requires_grad_(True)
            logits = model.classify_pre_bottleneck(adv)
            losses = F.cross_entropy(logits.float(), y, reduction="none")
            success = logits.detach().argmax(dim=-1).ne(y)
            with torch.no_grad():
                retained_loss = torch.maximum(retained_loss, losses.detach())
                replace = (success & ~best_success) | (success == best_success) & (
                    losses > best_loss
                )
                best, best_loss, best_success = _select(
                    best, best_loss, best_success, adv.detach(), losses.detach(), success
                )
                best_restart = torch.where(
                    replace, torch.full_like(best_restart, restart), best_restart
                )
            if step_index == attack_steps:
                break
            grad = torch.autograd.grad(losses.sum(), adv)[0]
            direction = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
            with torch.no_grad():
                adv = adv.detach() + direction * step.view(-1, 1)
                adv = pre + project_l2(adv - pre, radius)
    if torch.any(retained_loss + 1e-6 < initial_loss):
        raise RuntimeError(
            "Pre-quantization PGD highest retained loss is below its clean initial loss"
        )
    return AttackResult(best, best_loss, best_success, initial_loss, best_restart, retained_loss)
