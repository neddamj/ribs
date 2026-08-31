from types import SimpleNamespace

import torch

from ribs.collisions import targeted_collision_attack


class TinyVQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook = torch.nn.Parameter(torch.tensor([[0.0], [1.0]]))

    def forward(self, images, sample=False):
        pre = images.mean(dim=(1, 2, 3)).view(-1, 1, 1)
        indices = (pre[..., 0] > 0.5).long()
        quantized = self.codebook[indices]
        logits = torch.stack([images.mean((1, 2, 3)), -images.mean((1, 2, 3))], dim=1)
        return SimpleNamespace(
            canonical_latent=quantized.flatten(1),
            pre_bottleneck=pre,
            logits=logits,
            metadata={"code_indices": indices},
        )


class SourceReference(torch.nn.Module):
    def forward(self, images, sample=False):
        connected = images.mean((1, 2, 3)) * 0
        return SimpleNamespace(logits=torch.stack([connected + 1, connected], dim=1))


def test_vq_collision_attack_optimizes_prequantization_tokens():
    source = torch.full((1, 1, 2, 2), 0.9)
    target = torch.full((1, 1, 2, 2), 0.1)
    result = targeted_collision_attack(
        TinyVQ(),
        SourceReference(),
        source,
        torch.zeros(1, dtype=torch.long),
        target,
        epsilon=1.0,
        threshold=0.1,
        steps=5,
        restarts=1,
        lambda_sem=0.0,
    )
    assert result["criterion"] == "exact_vq"
    assert bool(result["successful"][0])
