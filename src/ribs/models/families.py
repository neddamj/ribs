"""Backbone and information-bottleneck model families."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import resnet18

from ..data import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class BottleneckOutput:
    logits: Tensor
    latent: Tensor
    canonical_latent: Tensor
    backbone_features: Tensor
    pre_bottleneck: Tensor
    aux_losses: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BottleneckModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def normalize_input(self, x: Tensor) -> Tensor:
        return (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)

    def encode_features(self, x: Tensor) -> Tensor:
        return self.backbone(self.normalize_input(x))

    def apply_bottleneck(self, features: Tensor, *, sample: bool = False) -> tuple[Tensor, dict]:
        raise NotImplementedError

    def encode(self, x: Tensor, *, sample: bool = False) -> BottleneckOutput:
        features = self.encode_features(x)
        latent, metadata = self.apply_bottleneck(features, sample=sample)
        return BottleneckOutput(
            logits=self.classify_latent(latent),
            latent=latent,
            canonical_latent=metadata.get("canonical_latent", latent),
            backbone_features=features,
            pre_bottleneck=metadata.get("pre_bottleneck", features),
            aux_losses=metadata.get("aux_losses", {}),
            metadata=metadata,
        )

    def classify_latent(self, z: Tensor) -> Tensor:
        return self.classifier(z)

    def forward(self, x: Tensor, *, sample: bool = False) -> BottleneckOutput:
        return self.encode(x, sample=sample)


class DimensionalBottleneck(BottleneckModel):
    def __init__(self, dz: int, num_classes: int = 10, identity: bool = False):
        super().__init__(num_classes)
        self.dz = dz
        self.identity = identity
        self.bottleneck = nn.Identity() if identity else nn.Linear(512, dz)
        self.classifier = nn.Linear(512 if identity else dz, num_classes)

    def apply_bottleneck(self, features: Tensor, *, sample: bool = False) -> tuple[Tensor, dict]:
        z = self.bottleneck(features)
        return z, {"canonical_latent": z, "pre_bottleneck": features}


class VIBBottleneck(BottleneckModel):
    def __init__(self, dz: int = 128, beta: float = 0.001, num_classes: int = 10):
        super().__init__(num_classes)
        self.dz = dz
        self.beta = beta
        self.fc_mu = nn.Linear(512, dz)
        self.fc_logvar = nn.Linear(512, dz)
        self.classifier = nn.Linear(dz, num_classes)

    def apply_bottleneck(self, features: Tensor, *, sample: bool = False) -> tuple[Tensor, dict]:
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if sample or self.training else mu
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()
        return z, {
            "canonical_latent": mu,
            "pre_bottleneck": features,
            "mu": mu,
            "logvar": logvar,
            "aux_losses": {"kl": kl},
        }


class VQBottleneck(BottleneckModel):
    def __init__(self, codebook_size: int = 128, num_classes: int = 10):
        super().__init__(num_classes)
        self.codebook_size = codebook_size
        self.tokens = 16
        self.token_dim = 32
        self.projection = nn.Linear(512, self.tokens * self.token_dim)
        bound = 1.0 / (self.token_dim**0.5)
        self.codebook = nn.Parameter(torch.empty(codebook_size, self.token_dim))
        nn.init.uniform_(self.codebook, -bound, bound)
        self.classifier = nn.Linear(self.tokens * self.token_dim, num_classes)

    def apply_bottleneck(self, features: Tensor, *, sample: bool = False) -> tuple[Tensor, dict]:
        z_e = self.projection(features).view(-1, self.tokens, self.token_dim)
        distances = (
            (z_e.unsqueeze(2) - self.codebook.view(1, 1, self.codebook_size, self.token_dim))
            .square()
            .sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
        z_q = self.codebook[indices]
        z_st = z_e + (z_q - z_e).detach()
        codebook_loss = (z_e.detach() - z_q).square().mean()
        commitment_loss = (z_e - z_q.detach()).square().mean()
        return z_st.flatten(1), {
            "canonical_latent": z_q.flatten(1),
            "pre_bottleneck": z_e,
            "code_indices": indices,
            "quantized_tokens": z_q,
            "aux_losses": {"codebook": codebook_loss, "commitment": 0.25 * commitment_loss},
        }

    def classify_pre_bottleneck(self, pre_bottleneck: Tensor) -> Tensor:
        z_e = pre_bottleneck.view(-1, self.tokens, self.token_dim)
        distances = (
            (z_e.unsqueeze(2) - self.codebook.view(1, 1, self.codebook_size, self.token_dim))
            .square()
            .sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
        z_q = self.codebook[indices]
        z_st = z_e + (z_q - z_e).detach()
        return self.classifier(z_st.flatten(1))


def quantize_uniform(z: Tensor, bits: int) -> Tensor:
    if bits < 1:
        raise ValueError("bits must be positive")
    levels = 2**bits - 1
    bounded = z.clamp(-1.0, 1.0)
    return 2.0 * torch.round((bounded + 1.0) * levels / 2.0) / levels - 1.0


class QuantizedContinuousBottleneck(BottleneckModel):
    def __init__(self, bits: int | None = 8, dz: int = 128, num_classes: int = 10):
        super().__init__(num_classes)
        self.bits = bits
        self.dz = dz
        self.projection = nn.Linear(512, dz)
        self.classifier = nn.Linear(dz, num_classes)

    def apply_bottleneck(self, features: Tensor, *, sample: bool = False) -> tuple[Tensor, dict]:
        continuous = torch.tanh(self.projection(features))
        if self.bits is None or self.bits >= 32:
            quantized = continuous
        else:
            quantized = quantize_uniform(continuous, self.bits)
        # Keep the exact quantized forward value while supplying the STE
        # derivative during attacks and training, including evaluation mode.
        latent = continuous + (quantized - continuous).detach()
        return latent, {
            "canonical_latent": quantized,
            "pre_bottleneck": continuous,
            "quantized_latent": quantized,
        }

    def classify_pre_bottleneck(self, pre_bottleneck: Tensor) -> Tensor:
        if self.bits is None or self.bits >= 32:
            quantized = pre_bottleneck
        else:
            quantized = quantize_uniform(pre_bottleneck, self.bits)
        latent = pre_bottleneck + (quantized - pre_bottleneck).detach()
        return self.classifier(latent)


class AutoencoderBottleneck(BottleneckModel):
    def __init__(self, dz: int = 128, num_classes: int = 10, image_size: int = 224):
        super().__init__(num_classes)
        self.dz = dz
        self.image_size = image_size
        self.decoder_base_size = max(1, image_size // 32)
        self.projection = nn.Linear(512, dz)
        blocks: list[nn.Module] = []
        channels = [512, 256, 128, 64, 32, 16]
        for in_channels, out_channels in pairwise(channels):
            groups = min(32, out_channels)
            blocks.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.GroupNorm(groups, out_channels),
                    nn.SiLU(),
                ]
            )
        self.decoder = nn.Sequential(
            nn.Linear(dz, 512 * self.decoder_base_size**2),
            nn.Unflatten(1, (512, self.decoder_base_size, self.decoder_base_size)),
            *blocks,
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def decode(self, z: Tensor) -> Tensor:
        reconstruction = self.decoder(z)
        if reconstruction.shape[-2:] != (self.image_size, self.image_size):
            reconstruction = torch.nn.functional.interpolate(
                reconstruction,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def classify_latent(self, z: Tensor) -> Tensor:
        raise RuntimeError(
            "Autoencoder classification requires a frozen reference classifier; "
            "use the autoencoder evaluation commands."
        )

    def apply_bottleneck(self, features: Tensor, *, sample: bool = False) -> tuple[Tensor, dict]:
        z = self.projection(features)
        reconstruction = self.decode(z)
        return z, {
            "canonical_latent": z,
            "pre_bottleneck": features,
            "reconstruction": reconstruction,
        }

    def forward(self, x: Tensor, *, sample: bool = False) -> BottleneckOutput:
        features = self.encode_features(x)
        z, metadata = self.apply_bottleneck(features, sample=sample)
        # Autoencoders are trained/evaluated through their reconstruction and a
        # separate frozen classifier; logits are intentionally unused.
        logits = torch.zeros(x.shape[0], self.num_classes, device=x.device, dtype=x.dtype)
        return BottleneckOutput(
            logits=logits,
            latent=z,
            canonical_latent=z,
            backbone_features=features,
            pre_bottleneck=features,
            metadata=metadata,
        )


def create_model(config: dict[str, Any]) -> BottleneckModel:
    model_config = config.get("model", config)
    family = model_config.get("family", "dimensional").lower()
    classes = int(model_config.get("num_classes", 10))
    if family == "dimensional":
        return DimensionalBottleneck(int(model_config.get("dz", 128)), classes)
    if family == "identity":
        return DimensionalBottleneck(512, classes, identity=True)
    if family == "vib":
        return VIBBottleneck(int(model_config.get("dz", 128)), float(model_config["beta"]), classes)
    if family == "vq":
        return VQBottleneck(int(model_config.get("codebook_size", 128)), classes)
    if family in {"quantized", "quantized_continuous"}:
        bits = model_config.get("bits", 8)
        bits = None if str(bits).lower() in {"fp32", "none"} else int(bits)
        return QuantizedContinuousBottleneck(bits, int(model_config.get("dz", 128)), classes)
    if family == "autoencoder":
        image_size = int(config.get("data", {}).get("image_size", 224))
        return AutoencoderBottleneck(int(model_config.get("dz", 128)), classes, image_size)
    raise ValueError(f"Unknown model family: {family}")
