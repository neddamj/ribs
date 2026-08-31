import pytest
import torch

from ribs.models import create_model


@pytest.mark.parametrize(
    "model_config",
    [
        {"family": "dimensional", "dz": 16},
        {"family": "vib", "dz": 8, "beta": 1e-3},
        {"family": "vq", "codebook_size": 8},
        {"family": "quantized", "dz": 8, "bits": 3},
        {"family": "autoencoder", "dz": 8},
    ],
)
def test_model_contract(model_config):
    model = create_model({"model": model_config, "data": {"image_size": 32}}).eval()
    output = model(torch.rand(2, 3, 32, 32), sample=False)
    assert output.logits.shape == (2, 10)
    assert output.latent.shape[0] == 2
    assert output.canonical_latent.shape[0] == 2
    assert output.backbone_features.shape == (2, 512)
    assert output.pre_bottleneck.shape[0] == 2
    if model_config["family"] == "autoencoder":
        assert output.metadata["reconstruction"].shape == (2, 3, 32, 32)


def test_quantizer_has_expected_levels():
    model = create_model({"model": {"family": "quantized", "dz": 4, "bits": 3}}).eval()
    output = model(torch.rand(16, 3, 32, 32))
    assert output.canonical_latent.min() >= -1
    assert output.canonical_latent.max() <= 1
    assert len(torch.unique(output.canonical_latent)) <= 2**3


def test_raw_pixel_epsilon_is_applied_before_normalization():
    model = create_model({"model": {"family": "dimensional", "dz": 4}}).eval()
    raw = torch.full((1, 3, 2, 2), 0.5)
    epsilon = 1 / 255
    normalized_delta = model.normalize_input(raw + epsilon) - model.normalize_input(raw)
    expected = torch.full_like(raw, epsilon) / model.image_std
    assert torch.allclose(normalized_delta, expected, atol=1e-7)


def test_dimensional_model_overfits_fixed_batch():
    torch.manual_seed(0)
    model = create_model({"model": {"family": "dimensional", "dz": 16}})
    model.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 512))
    labels = torch.arange(32) % 10
    images = torch.zeros(32, 3, 4, 4)
    images.flatten(1)[torch.arange(32), labels] = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images).logits, labels)
        loss.backward()
        optimizer.step()
    assert (model(images).logits.argmax(1) == labels).float().mean() == 1.0
