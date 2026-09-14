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
    _cache_collision_images,
    _collision_batch_sizes,
    _eligibility_records,
    _ReconstructionTask,
    _resumable_evaluation_dir,
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


def test_load_model_freezes_parameters_without_disabling_input_gradients(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run" / "checkpoints" / "best_tune_accuracy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    model = torch.nn.Linear(2, 2, bias=False)
    state = {"config": {"device": "cpu"}, "model": model.state_dict()}
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        evaluation, "create_model", lambda config: torch.nn.Linear(2, 2, bias=False)
    )
    monkeypatch.setattr(evaluation, "choose_device", lambda config: torch.device("cpu"))

    loaded, _, _ = evaluation.load_model(checkpoint.parent.parent)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    inputs = torch.ones(1, 2, requires_grad=True)
    loaded(inputs).sum().backward()
    assert inputs.grad is not None


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


def test_collision_eligibility_is_independent_of_attack_batch_size():
    assert _collision_batch_sizes({"evaluation_batch_size": 32, "collision_batch_size": 8}) == (
        8,
        32,
    )
    assert _collision_batch_sizes(
        {"evaluation_batch_size": 32, "collision_batch_size": 8}, "autoencoder"
    ) == (1, 32)
    with pytest.raises(ValueError, match="must be positive"):
        _collision_batch_sizes({"evaluation_batch_size": 0, "collision_batch_size": 8})


def test_collision_image_cache_decodes_each_selected_index_once():
    class CountingDataset:
        def __init__(self):
            self.calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            return {"image": torch.tensor([float(index)])}

    dataset = CountingDataset()
    cache = _cache_collision_images(dataset, [(2, 4), (2, 7), (4, 7)])
    assert dataset.calls == [2, 4, 7]
    assert sorted(cache) == [2, 4, 7]


def test_collision_eligibility_is_batch_partition_independent():
    model = FakeAttackModel()
    reference = FakeAttackModel()
    full = {
        "image": torch.tensor([[[[0.9, 0.1]]], [[[0.1, 0.9]]]]),
        "label": torch.tensor([0, 1]),
        "sample_id": ["a", "b"],
    }
    split = [{key: value[index : index + 1] for key, value in full.items()} for index in range(2)]
    full_result = _eligibility_records(model, reference, [full], torch.device("cpu"))
    split_result = _eligibility_records(model, reference, split, torch.device("cpu"))
    assert full_result[0] == split_result[0]
    assert torch.equal(full_result[1], split_result[1])
    assert torch.equal(full_result[2], split_result[2])


def test_resumable_evaluation_reuses_only_matching_attempt(tmp_path):
    run_dir = tmp_path / "run"
    config = {"kind": "collision", "split": "final", "seed": 2025}
    evaluation_dir, complete = _resumable_evaluation_dir(
        run_dir, "collision", config, "collision_attacks.parquet"
    )
    assert not complete
    resumed, complete = _resumable_evaluation_dir(
        run_dir, "collision", config, "collision_attacks.parquet"
    )
    assert resumed == evaluation_dir
    assert not complete

    pd.DataFrame({"value": [1]}).to_parquet(
        evaluation_dir / "collision_attacks.parquet", index=False
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n")
    resumed, complete = _resumable_evaluation_dir(
        run_dir, "collision", config, "collision_attacks.parquet"
    )
    assert resumed == evaluation_dir
    assert complete


def test_collision_evaluation_resumes_completed_shards(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    reference_dir = tmp_path / "reference"
    for directory in (run_dir, reference_dir):
        checkpoint = directory / "checkpoints" / "best_tune_accuracy.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    pd.DataFrame({"label": [0, 0, 1, 1]}).to_parquet(
        artifacts / "latents_development_tune_index.parquet", index=False
    )

    class CollisionModel(torch.nn.Module):
        def forward(self, images, sample=False):
            latent = images.flatten(1).mean(1, keepdim=True)
            logits = torch.stack([latent[:, 0] + 1, latent[:, 0]], dim=1)
            return SimpleNamespace(canonical_latent=latent, logits=logits, metadata={})

    class Dataset:
        def __getitem__(self, index):
            return {"image": torch.full((1, 2, 2), float(index))}

    config = {
        "seed": 0,
        "model": {"family": "dimensional"},
        "attack": {
            "seed": 2025,
            "evaluation_batch_size": 2,
            "collision_batch_size": 1,
            "input_epsilons": [0.0, 0.1],
            "collision_steps": 1,
            "collision_restarts": 1,
        },
    }
    reference_config = {"seed": 0, "model": {"family": "identity"}}
    model = CollisionModel()
    reference = CollisionModel()
    load_calls = 0

    def fake_load_model(path, checkpoint=None):
        nonlocal load_calls
        load_calls += 1
        selected = reference if Path(path) == reference_dir else model
        selected_config = reference_config if Path(path) == reference_dir else config
        return selected, selected_config, torch.device("cpu")

    monkeypatch.setattr(evaluation, "load_model", fake_load_model)
    monkeypatch.setattr(evaluation, "_validate_reference_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation,
        "make_loader",
        lambda *args, **kwargs: SimpleNamespace(dataset=Dataset()),
    )
    monkeypatch.setattr(
        evaluation,
        "_eligibility_records",
        lambda *args: (["source", "target"], torch.tensor([0, 1]), torch.tensor([True, True])),
    )
    monkeypatch.setattr(evaluation, "_candidate_collision_pairs", lambda *args: [(0, 1)])
    monkeypatch.setattr(evaluation, "extract_latents", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation,
        "load_latents",
        lambda *args: {"canonical_latent": torch.tensor([[0.0], [0.1], [0.9], [1.0]])},
    )

    attempted_epsilons = []
    fail_once = True

    def fake_attack(model, reference, source, labels, target, epsilon, *args, **kwargs):
        nonlocal fail_once
        attempted_epsilons.append(epsilon)
        if epsilon == 0.1 and fail_once:
            fail_once = False
            raise RuntimeError("simulated interruption")
        return {
            "adversarial": source,
            "successful": torch.zeros(len(source), dtype=torch.bool),
            "criterion": "threshold",
        }

    monkeypatch.setattr(evaluation, "targeted_collision_attack", fake_attack)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluation.evaluate_collision_attacks(
            run_dir, reference_dir, split="development_tune", max_pairs=1, lambda_sem=1.0
        )

    result = evaluation.evaluate_collision_attacks(
        run_dir, reference_dir, split="development_tune", max_pairs=1, lambda_sem=1.0
    )
    assert result.is_file()
    assert attempted_epsilons == [0.0, 0.1, 0.1]
    assert len(pd.read_parquet(result)) == 2
    assert load_calls == 4


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
