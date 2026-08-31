import pandas as pd
import torch

from ribs.artifacts import load_latents, save_frame, save_latents


def test_machine_readable_artifacts_round_trip(tmp_path):
    frame = pd.DataFrame({"sample_id": ["a", "b"], "metric": [0.25, 0.75]})
    frame_path = tmp_path / "sample_metrics.parquet"
    save_frame(frame, frame_path)
    pd.testing.assert_frame_equal(pd.read_parquet(frame_path), frame)

    tensors = {"canonical_latent": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    latent_path = tmp_path / "latents.safetensors"
    save_latents(latent_path, tensors)
    assert torch.equal(load_latents(latent_path)["canonical_latent"], tensors["canonical_latent"])
