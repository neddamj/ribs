import sys
from types import SimpleNamespace

import torch

from ribs.attacks import input_pgd, latent_pgd, project_l2
from ribs.square_attack import square_attack


class ToyModel(torch.nn.Module):
    def forward(self, x, sample=False):
        latent = x.flatten(1)
        logits = torch.cat([latent[:, :1], -latent[:, :1]], dim=1)
        return SimpleNamespace(logits=logits, latent=latent, canonical_latent=latent)

    def classify_latent(self, latent):
        return torch.cat([latent[:, :1], -latent[:, :1]], dim=1)


def test_input_pgd_respects_bound_and_zero_radius():
    model = ToyModel()
    x = torch.full((4, 1, 2, 2), 0.75)
    y = torch.zeros(4, dtype=torch.long)
    zero = input_pgd(model, x, y, 0.0, steps=2, restarts=1)
    assert torch.equal(zero.adversarial, x)
    result = input_pgd(model, x, y, 0.1, steps=2, restarts=2)
    assert (result.adversarial - x).abs().max() <= 0.100001
    assert result.adversarial.min() >= 0 and result.adversarial.max() <= 1
    assert torch.all(result.loss >= result.initial_loss - 1e-6)


def test_latent_projection_is_per_sample():
    delta = torch.randn(3, 4)
    radius = torch.tensor([0.1, 1.0, 2.0])
    projected = project_l2(delta, radius)
    assert torch.all(projected.norm(dim=1) <= radius + 1e-6)


def test_latent_pgd_returns_latent_shape():
    model = ToyModel()
    x = torch.full((2, 1, 2, 2), 0.75)
    y = torch.zeros(2, dtype=torch.long)
    result = latent_pgd(model, x, y, 0.1, steps=2, restarts=1)
    assert result.adversarial.shape == (2, 4)


def test_square_attack_respects_bound(monkeypatch):
    class FakeSquare:
        def __init__(self, model, eps, **kwargs):
            self.eps = eps

        def __call__(self, images, labels):
            return (images - self.eps).clamp(0, 1)

    monkeypatch.setitem(sys.modules, "torchattacks", SimpleNamespace(Square=FakeSquare))
    model = ToyModel()
    x = torch.full((2, 1, 4, 4), 0.75)
    y = torch.zeros(2, dtype=torch.long)
    result = square_attack(model, x, y, 0.1, queries=3)
    assert (result.adversarial - x).abs().max() <= 0.100001
