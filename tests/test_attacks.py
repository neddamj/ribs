import sys
from types import SimpleNamespace

import torch

from ribs.attacks import _uniform_l2_noise, input_pgd, latent_pgd, project_l2
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


def test_input_pgd_assesses_and_retains_the_final_update():
    model = ToyModel()
    x = torch.full((4, 1, 2, 2), 0.75)
    y = torch.zeros(4, dtype=torch.long)
    result = input_pgd(model, x, y, 0.1, steps=1, restarts=1, seed=12)
    assert torch.allclose(result.adversarial[:, 0, 0, 0], torch.full((4,), 0.65))


def test_input_pgd_eot_uses_exact_sample_count_without_graph_accumulation():
    model = ToyModel()
    x = torch.full((2, 1, 2, 2), 0.75)
    y = torch.zeros(2, dtype=torch.long)
    result = input_pgd(model, x, y, 0.1, steps=1, restarts=1, eot_samples=4)
    assert torch.isfinite(result.loss).all()
    assert (result.adversarial - x).abs().max() <= 0.100001


def test_input_pgd_can_carry_a_smaller_radius_candidate_forward():
    model = ToyModel()
    x = torch.full((2, 1, 2, 2), 0.75)
    y = torch.zeros(2, dtype=torch.long)
    candidate = x - 0.05
    result = input_pgd(
        model,
        x,
        y,
        0.1,
        steps=1,
        restarts=0,
        seed=1,
        initial_adversarial=candidate,
    )
    assert torch.equal(result.adversarial, candidate)
    assert result.restart.tolist() == [-2, -2]


def test_latent_projection_is_per_sample():
    delta = torch.randn(3, 4)
    radius = torch.tensor([0.1, 1.0, 2.0])
    projected = project_l2(delta, radius)
    assert torch.all(projected.norm(dim=1) <= radius + 1e-6)


def test_latent_initialization_is_inside_each_l2_ball():
    reference = torch.zeros(128, 8)
    radii = torch.linspace(0.1, 1.0, len(reference))
    generator = torch.Generator().manual_seed(9)
    noise = _uniform_l2_noise(reference, radii, generator)
    norms = noise.norm(dim=1)
    assert torch.all(norms <= radii + 1e-6)
    assert torch.all(norms > 0)


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
