from types import SimpleNamespace

import torch

from ribs.invariance import invariance_metrics


class ScaleEncoder(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, images, sample=False):
        latent = images.flatten(1) * self.scale
        score = images.flatten(1).mean(1)
        logits = torch.stack([score, -score], dim=1)
        return SimpleNamespace(logits=logits, canonical_latent=latent)


def test_invariance_uses_unit_normalized_latents_and_reports_accuracy():
    images = torch.rand(2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)
    sample_ids = ["a", "b"]
    first = invariance_metrics(ScaleEncoder(1.0), images, labels, sample_ids)
    second = invariance_metrics(ScaleEncoder(100.0), images, labels, sample_ids)
    assert abs(first["nuisance_distance_macro"] - second["nuisance_distance_macro"]) < 1e-6
    assert "accuracy_translation" in first
    assert "accuracy_horizontal_flip" in first
