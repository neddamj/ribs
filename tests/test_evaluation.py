from types import SimpleNamespace

import pandas as pd
import torch

from ribs.analysis import geometry_correlations
from ribs.evaluation import _ReconstructionTask


class FakeAutoencoder(torch.nn.Module):
    def forward(self, x, sample=False):
        z = x.flatten(1)[:, :2]
        return SimpleNamespace(metadata={"reconstruction": x}, latent=z, canonical_latent=z)

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, 2, 2)


class FakeReference(torch.nn.Module):
    def forward(self, x, sample=False):
        return SimpleNamespace(logits=torch.zeros(x.shape[0], 10))


def test_reconstruction_task_exposes_logits_and_latent_classifier():
    task = _ReconstructionTask(FakeAutoencoder(), FakeReference())
    output = task(torch.rand(2, 3, 2, 2))
    assert output.logits.shape == (2, 10)
    assert task.classify_latent(torch.rand(2, 2)).shape == (2, 10)


def test_geometry_correlations_produces_monotone_bh_q_values():
    frame = pd.DataFrame(
        {
            "robust_auc": [1, 2, 3, 4, 5, 6],
            "intra_class_distance": [1, 2, 3, 4, 5, 6],
            "inter_class_distance": [1, 3, 2, 6, 4, 5],
            "separation_ratio": [6, 1, 5, 2, 4, 3],
        }
    )
    result = geometry_correlations(frame).sort_values("p_value")
    assert result.bh_q_value.is_monotonic_increasing
