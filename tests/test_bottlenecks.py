import torch

from ribs.attacks import prequantization_latent_pgd
from ribs.models.families import VIBBottleneck, VQBottleneck, quantize_uniform


def test_vib_kl_matches_hand_computation():
    model = VIBBottleneck(dz=2, beta=1e-3)
    features = torch.zeros(1, 512)
    with torch.no_grad():
        model.fc_mu.weight.zero_()
        model.fc_mu.bias.copy_(torch.tensor([1.0, 2.0]))
        model.fc_logvar.weight.zero_()
        model.fc_logvar.bias.zero_()
    _, metadata = model.apply_bottleneck(features, sample=False)
    expected = 0.5 * (1.0**2 + 2.0**2)
    assert torch.allclose(metadata["aux_losses"]["kl"], torch.tensor(expected))


def test_vib_kl_is_zero_for_standard_normal_parameters():
    model = VIBBottleneck(dz=2, beta=1e-3)
    features = torch.zeros(1, 512)
    with torch.no_grad():
        model.fc_mu.weight.zero_()
        model.fc_mu.bias.zero_()
        model.fc_logvar.weight.zero_()
        model.fc_logvar.bias.zero_()
    _, metadata = model.apply_bottleneck(features, sample=False)
    assert metadata["aux_losses"]["kl"].item() == 0.0


def test_vq_selects_nearest_code_and_has_ste_gradient():
    model = VQBottleneck(codebook_size=2)
    features = torch.zeros(1, 512, requires_grad=True)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.fill_(0.1)
        model.codebook[0].fill_(0.0)
        model.codebook[1].fill_(1.0)
    latent, metadata = model.apply_bottleneck(features)
    assert torch.all(metadata["code_indices"] == 0)
    latent.sum().backward()
    assert features.grad is not None


def test_uniform_quantizer_is_bounded_and_discrete():
    values = torch.linspace(-1, 1, 100)
    quantized = quantize_uniform(values, 2)
    assert quantized.min() >= -1 and quantized.max() <= 1
    assert len(torch.unique(quantized)) <= 4


def test_prequantization_attack_uses_family_surface():
    model = VQBottleneck(codebook_size=4).eval()
    x = torch.rand(2, 3, 32, 32)
    y = torch.zeros(2, dtype=torch.long)
    result = prequantization_latent_pgd(model, x, y, 0.1, steps=1, restarts=1)
    assert result.adversarial.ndim == 2
    assert result.adversarial.shape[1] == 16 * 32
