import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from ribs import evaluation
from ribs.analysis import geometry_correlations
from ribs.evaluation import (
    _collision_batch_sizes,
    _ReconstructionTask,
    select_collision_lambda,
    validate_nearest_capacity_source,
)


class FakeAutoencoder(torch.nn.Module):
    def forward(self, x, sample=False):
        z = x.flatten(1)[:, :2]
        return SimpleNamespace(metadata={"reconstruction": x}, latent=z, canonical_latent=z)

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, 2, 2)


class FakeReference(torch.nn.Module):
    def forward(self, x, sample=False):
        return SimpleNamespace(logits=torch.zeros(x.shape[0], 10))


class FakeAttackModel(torch.nn.Module):
    def forward(self, x, sample=False):
        latent = x.flatten(1)[:, :2]
        return SimpleNamespace(
            logits=self.classify_latent(latent),
            canonical_latent=latent,
        )

    def classify_latent(self, latent):
        return latent


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


def test_attack_evaluation_sets_eval_mode_before_clean_prediction(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best_tune_accuracy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    model = FakeAttackModel().train()
    config = {
        "seed": 0,
        "model": {"family": "dimensional"},
        "train": {"batch_size": 2},
        "attack": {
            "seed": 2025,
            "input_epsilons": [0.0],
            "latent_rhos": [],
            "steps": 1,
            "restarts": 1,
            "run_diagnostics": False,
        },
    }
    batch = {
        "image": torch.tensor([[[[0.9, 0.1]]], [[[0.1, 0.9]]]]),
        "label": torch.tensor([0, 1]),
        "sample_id": ["a", "b"],
    }
    monkeypatch.setattr(
        evaluation, "load_model", lambda run_dir, checkpoint: (model, config, torch.device("cpu"))
    )
    loader_arguments = {}

    def fake_loader(*args, **kwargs):
        loader_arguments.update(kwargs)
        return [batch]

    monkeypatch.setattr(evaluation, "make_loader", fake_loader)
    predict_logits = evaluation._predict_logits

    def assert_eval_mode(model, images, samples, seed=None):
        assert not model.training
        return predict_logits(model, images, samples, seed)

    monkeypatch.setattr(evaluation, "_predict_logits", assert_eval_mode)

    paths = evaluation.evaluate_attacks(run_dir, max_samples=2)

    input_records = pd.read_parquet(paths["input"])
    assert input_records.clean_prediction.tolist() == [0, 1]
    assert loader_arguments["batch_size"] == 2


def test_collision_lambda_selection_is_deterministic_and_prespecified():
    metrics = pd.DataFrame(
        {
            "lambda_sem": [3.0, 1.0, 0.3],
            "collision_success_rate": [0.5, 0.5, 0.4],
            "reference_source_preservation_rate": [0.8, 0.9, 1.0],
            "median_distance": [0.1, 0.2, 0.05],
        }
    )
    selection = select_collision_lambda(metrics)
    assert selection["selected_lambda_sem"] == 1.0


def test_collision_eligibility_uses_memory_safe_batch_cap():
    assert _collision_batch_sizes({"evaluation_batch_size": 32, "collision_batch_size": 8}) == (
        8,
        8,
    )
    assert _collision_batch_sizes(
        {"evaluation_batch_size": 32, "collision_batch_size": 8}, "autoencoder"
    ) == (1, 8)
    with pytest.raises(ValueError, match="must be positive"):
        _collision_batch_sizes({"evaluation_batch_size": 0, "collision_batch_size": 8})


def test_transfer_source_is_selected_by_nearest_same_seed_effective_rank(tmp_path):
    def make_run(family, name, rank):
        run = tmp_path / family / name
        analysis = run / "analysis" / "geometry-fixture-attempt0"
        analysis.mkdir(parents=True)
        (run / "COMPLETED").write_text("completed\n")
        (run / "data_manifest_hash.txt").write_text("shared\n")
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump({"seed": 1, "model": {"family": family}})
        )
        (analysis / "config.json").write_text(json.dumps({"split": "final", "max_samples": None}))
        (analysis / "geometry.json").write_text(json.dumps({"effective_rank": rank}))
        (analysis / "COMPLETED").write_text("completed\n")
        return run

    target = make_run("vq", "target", 5.0)
    nearest = make_run("dimensional", "near", 4.8)
    farther = make_run("vib", "far", 6.0)
    config = {"seed": 1, "model": {"family": "vq"}}

    selected = validate_nearest_capacity_source(target, None, config)
    assert Path(selected["matched_source_run"]) == nearest.resolve()
    with pytest.raises(ValueError, match="not the nearest"):
        validate_nearest_capacity_source(target, farther, config)
