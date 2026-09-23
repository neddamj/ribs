import json
from pathlib import Path

import pandas as pd
import yaml

from ribs.analysis import (
    _completed_analysis_file,
    _new_analysis_dir,
    aggregate_completed_runs,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_aggregate_completed_run_fixture(tmp_path):
    run_dir = tmp_path / "dimensional" / "run"
    evaluation_dir = run_dir / "evaluations" / "robustness-fixture-attempt0"
    evaluation_dir.mkdir(parents=True)
    config = {
        "seed": 0,
        "model": {"family": "dimensional", "dz": 512},
        "attack": {"input_epsilons": [0.0, 1.0], "latent_rhos": [1.0]},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "COMPLETED").write_text("completed\n")
    (run_dir / "metrics.json").write_text(json.dumps({"codebook_collapsed": False}))
    (evaluation_dir / "config.json").write_text(
        json.dumps({"split": "final", "max_samples": None, "attack_protocol_version": 2})
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n")
    records = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "radius": [0.0, 1.0, 0.0, 1.0],
            "clean_correct": [True, True, True, True],
            "successful": [False, True, False, False],
        }
    )
    records.to_parquet(evaluation_dir / "input_pgd.parquet", index=False)
    records.to_parquet(evaluation_dir / "latent_pgd.parquet", index=False)

    summary = aggregate_completed_runs(tmp_path)

    assert set(summary.attack) == {"input_pgd", "latent_pgd"}
    assert (tmp_path / "phase1_hierarchical_bootstrap.parquet").is_file()


def test_phase2_aggregate_is_namespaced_and_records_strength(tmp_path):
    run_dir = tmp_path / "vib" / "run"
    evaluation_dir = run_dir / "evaluations" / "robustness-fixture-attempt0"
    evaluation_dir.mkdir(parents=True)
    config = {
        "seed": 0,
        "model": {"family": "vib", "dz": 128, "beta": 0.0001},
        "attack": {"input_epsilons": [0.0, 1.0], "latent_rhos": [1.0]},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "COMPLETED").write_text("completed\n")
    (run_dir / "metrics.json").write_text(json.dumps({"codebook_collapsed": False}))
    (evaluation_dir / "config.json").write_text(
        json.dumps({"split": "final", "max_samples": None, "attack_protocol_version": 2})
    )
    (evaluation_dir / "COMPLETED").write_text("completed\n")
    records = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "radius": [0.0, 1.0, 0.0, 1.0],
            "clean_correct": [True, True, True, True],
            "successful": [False, True, False, False],
        }
    )
    records.to_parquet(evaluation_dir / "input_pgd.parquet", index=False)
    records.to_parquet(evaluation_dir / "latent_pgd.parquet", index=False)

    aggregate_completed_runs(tmp_path, output_prefix="phase2", include_families=["vib"])

    summary = pd.read_parquet(tmp_path / "phase2_summary.parquet")
    assert summary.strength_parameter.iloc[0] == "beta"
    assert summary.strength_value.iloc[0] == "0.0001"
    assert summary.bottleneck_strength_ordinal.iloc[0] == 1
    assert (tmp_path / "phase2_seed_summary.parquet").is_file()
    bootstrap = pd.read_parquet(tmp_path / "phase2_hierarchical_bootstrap.parquet")
    assert "beta" in bootstrap.columns
    assert set(bootstrap.beta.dropna()) == {0.0001}
    assert not (tmp_path / "phase1_summary.parquet").exists()


def test_analysis_attempts_are_immutable_and_truncated_attempts_are_not_final(tmp_path):
    run_dir = tmp_path / "run"
    partial_config = {"split": "final", "max_samples": 2}
    partial = _new_analysis_dir(run_dir, "geometry", partial_config)
    (partial / "config.json").write_text(json.dumps(partial_config))
    (partial / "geometry.json").write_text("{}")
    (partial / "COMPLETED").write_text("completed\n")
    full_config = {"split": "final", "max_samples": None}
    full = _new_analysis_dir(run_dir, "geometry", full_config)
    (full / "config.json").write_text(json.dumps(full_config))
    (full / "geometry.json").write_text("{}")
    (full / "COMPLETED").write_text("completed\n")

    selected = _completed_analysis_file(
        run_dir, "geometry", "final", "geometry.json", "geometry_final.json"
    )

    assert selected == full / "geometry.json"
    assert partial != full


def test_phase2_amendment_keeps_omitted_runs_in_representation_summary(tmp_path):
    run_dir = tmp_path / "vib" / "run"
    run_dir.mkdir(parents=True)
    config = {
        "seed": 0,
        "model": {"family": "vib", "dz": 128, "beta": 0.0001},
        "attack": {"input_epsilons": [0.0, 1.0], "latent_rhos": [1.0]},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "COMPLETED").write_text("completed\n")
    (run_dir / "metrics.json").write_text(json.dumps({"codebook_collapsed": False}))

    summary = aggregate_completed_runs(
        tmp_path,
        output_prefix="phase2",
        include_families=["vib"],
        runtime_amendment=PROJECT_ROOT / "configs" / "phase2_runtime_amendment_20260917.yaml",
    )

    representation = pd.read_parquet(tmp_path / "phase2_representation_summary.parquet")
    assert summary.empty
    assert representation.run_dir.tolist() == [str(run_dir)]
    assert representation.strength_value.tolist() == ["0.0001"]
