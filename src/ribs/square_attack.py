"""Adapter for the maintained torchattacks Square Attack implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attacks import AttackResult


class _LogitsAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images, sample=False).logits


@torch.no_grad()
def square_attack(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    epsilon: float,
    queries: int = 5000,
    seed: int = 0,
) -> AttackResult:
    """Run the independently maintained Square Attack with an L-infinity bound."""
    model.eval()
    initial_logits = model(x, sample=False).logits
    initial_loss = F.cross_entropy(initial_logits.float(), y, reduction="none")
    best_x = x.detach().clone()
    best_loss = initial_loss.clone()
    best_success = initial_logits.argmax(-1).ne(y)
    if epsilon == 0 or queries <= 0:
        return AttackResult(best_x, best_loss, best_success, initial_loss)
    try:
        import torchattacks
    except ImportError as exc:
        raise RuntimeError(
            "Square Attack requires the pinned torchattacks dependency; "
            "install the locked environment"
        ) from exc
    adapter = _LogitsAdapter(model).to(x.device).eval()
    attack = torchattacks.Square(
        adapter,
        norm="Linf",
        eps=epsilon,
        n_queries=queries,
        n_restarts=1,
        p_init=0.8,
        loss="margin",
        resc_schedule=True,
        seed=seed,
        verbose=False,
    )
    candidate = attack(x, y).detach()
    if (candidate - x).abs().flatten(1).amax(1).max() > epsilon + 1e-6:
        raise RuntimeError("Square Attack returned an example outside the configured bound")
    if candidate.min() < -1e-6 or candidate.max() > 1.0 + 1e-6:
        raise RuntimeError("Square Attack returned pixels outside [0, 1]")
    logits = model(candidate, sample=False).logits
    losses = F.cross_entropy(logits.float(), y, reduction="none")
    success = logits.argmax(-1).ne(y)
    replace = (success & ~best_success) | (success == best_success) & (losses > best_loss)
    best_x = torch.where(replace.view(-1, *([1] * (x.ndim - 1))), candidate, best_x)
    best_loss = torch.where(replace, losses, best_loss)
    best_success = torch.where(replace, success, best_success)
    return AttackResult(best_x, best_loss, best_success, initial_loss)
