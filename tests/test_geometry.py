import torch

from ribs.geometry import effective_rank, pairwise_class_distances


def test_blockwise_distances_match_direct_matrix():
    latents = torch.tensor([[1.0, 0], [0.0, 1], [1.0, 1], [-1.0, 0]])
    labels = torch.tensor([0, 0, 1, 1])
    intra, inter, nearest = pairwise_class_distances(latents, labels, block_size=2)
    normalized = latents / latents.norm(dim=1, keepdim=True)
    distance = torch.cdist(normalized, normalized)
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    assert abs(intra - float(distance[same].mean())) < 1e-6
    assert abs(inter - float(distance[~(labels[:, None] == labels[None, :])].mean())) < 1e-6
    assert torch.allclose(
        nearest,
        distance.masked_fill(labels[:, None] == labels[None, :], float("inf"))
        .min(1)
        .values.double(),
    )


def test_effective_rank_is_bounded():
    rank, eigenvalues = effective_rank(torch.randn(20, 5))
    assert 1 <= rank <= 5
    assert eigenvalues.ndim == 1


def test_pairwise_distances_weight_each_anchor_equally_for_unbalanced_classes():
    latents = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 0, 1, 2])
    intra, inter, _ = pairwise_class_distances(latents, labels, block_size=2)
    normalized = latents / latents.norm(dim=1, keepdim=True)
    distances = torch.cdist(normalized, normalized)
    intra_anchors = []
    inter_anchors = []
    for index in range(len(labels)):
        same = (labels == labels[index]) & (torch.arange(len(labels)) != index)
        different = labels != labels[index]
        if same.any():
            intra_anchors.append(float(distances[index, same].mean()))
        inter_anchors.append(float(distances[index, different].mean()))
    assert abs(intra - sum(intra_anchors) / len(intra_anchors)) < 1e-6
    assert abs(inter - sum(inter_anchors) / len(inter_anchors)) < 1e-6
